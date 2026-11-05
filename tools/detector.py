"""Motion detector for WiFi Space Mapper — the proven pipeline as a reusable class.

This is the exact detector that live_csi_plot.py validated on hardware, lifted out
of the plotting loop so both the simple matplotlib viewer and the PyQtGraph
visualizer share one implementation (and so it can be unit-tested headlessly).

Why each stage exists (each defeats a specific way the naive "threshold the
variance" detector fails):

  1. Hampel filter — removes spike outliers before any computation so a single bad
     sample cannot inflate the moving-window std or masquerade as sustained motion.
  2. Guard/DC masking — excludes known-irrelevant subcarriers (guard bands + DC)
     from all variance computations, preventing silent degradation from noise bins.
  3. Gain removal — divide each frame by its own mean. The ESP32 AGC scales the
     whole amplitude vector per packet; dividing it out leaves only the channel
     shape, so gain wobble doesn't look like motion. (The #1 false-positive source.)
  4. Moving-window std (not frame-diff) — over a ~2 s window, measure how much each
     subcarrier's normalized amplitude churns. Still -> near-flat -> low; motion -> high.
  5. Calibrated threshold — learn the quiet "still" level for ~8 s at startup and FIX
     the threshold at P95(baseline) x 1.4, so continuous motion can't drag it upward.
  6. Hysteresis — separate enter/exit levels so the MOTION/STILL readout does not chatter.

Rate independence: the window and calibration are defined in SECONDS and converted to
frame counts using the fps measured live during warmup, so the detector behaves the
same at 22 fps or 90 fps. Measuring fps honestly needs two tricks: skip the sparse
boot/Wi-Fi-reconnect ramp, then use the MEDIAN inter-frame interval (USB delivers
frames in ~16 ms bursts, so a mean count/elapsed estimate lies).

Usage:
    det = MotionDetector()
    for frame in frames:
        st = det.update(frame.amp, frame.t)
        # st.phase in {"warmup","calibrate","detect"}; st.in_motion is the answer
"""

import time
from collections import deque, Counter
from dataclasses import dataclass
from typing import Optional

import numpy as np

# --- RuView port modules ---
# component_4 is the single source of truth for guard/DC masks and the frame
# quality scorer; subcarrier_select ranks bins by motion sensitivity.
# detector.py deliberately keeps no local copies of either.
from component_4 import combined_mask
from subcarrier_select import (
    select_sensitive_subcarriers,
    select_by_variance,
    SubcarrierSelection,
)

# Component 2 tuning (variance-ratio selection during adaptation)
SEL_TOP_K = 24           # subcarriers kept for the motion metric
SEL_MIN_SENSITIVITY = 1.5  # minimum motion-var/static-var ratio to qualify
SEL_RATE = 0.35          # measurable improvement threshold for the refactor ratio


def _median_interval_fps(times):
    """Estimate fps from timestamps using the median inter-frame interval.

    Immune to the USB burst inflation that makes a mean count/elapsed estimate
    overestimate the rate and stretch every time constant. See CHANGELOG.md
    section Rate-aware.
    """
    if len(times) < 2:
        return 0.0
    tl = list(times)
    intervals = sorted(tl[i + 1] - tl[i] for i in range(len(tl) - 1))
    intervals = [t for t in intervals if t > 1e-6]
    if not intervals:
        return 0.0
    return 1.0 / intervals[len(intervals) // 2]


# Defaults are the hardware-validated values from live_csi_plot.py.
HAMPHALF = 5      # Hampel half-window (window = 2*HAMPHALF+1 frames)
HAMPTHR = 3.0     # Hampel MAD-scale threshold
WARMUP_SETTLE_FPS = 15     # stream is up once it exceeds this (skips boot ramp)
WARMUP_MEASURE_SEC = 3.0   # then measure the true fps over this window
WINDOW_SEC = 2.0           # moving-std window duration
CALIB_SEC = 8.0            # still-baseline duration (STAY STILL this long)
THRESH_MULT = 1.4          # threshold = P95(still motion) * this
HYST_LOW = 0.6             # exit-motion level = HYST_LOW * threshold


@dataclass
class DetectorState:
    """A snapshot returned by every update — everything a UI needs to render."""
    phase: str                       # warmup | calibrate | detect
    in_motion: bool                  # the MOTION/STILL answer (detect phase)
    motion_level: Optional[float]    # current moving-window churn, or None until the window fills
    threshold: Optional[float]       # fixed once calibrated
    fps: Optional[float]             # measured during warmup
    target_len: Optional[int]        # locked subcarrier count
    calib_progress: float            # 0..1 during calibration


class MotionDetector:
    def __init__(self, window_sec=WINDOW_SEC, calib_sec=CALIB_SEC,
                 thresh_mult=THRESH_MULT, hyst_low=HYST_LOW,
                 warmup_settle_fps=WARMUP_SETTLE_FPS,
                 warmup_measure_sec=WARMUP_MEASURE_SEC):
        self.window_sec = window_sec
        self.calib_sec = calib_sec
        self.thresh_mult = thresh_mult
        self.hyst_low = hyst_low
        self.warmup_settle_fps = warmup_settle_fps
        self.warmup_measure_sec = warmup_measure_sec

        # --- warmup state ---
        self.phase = "warmup"
        self._len_votes = Counter()
        self._fps_times = deque(maxlen=240)
        self._measure_start = None
        self._measure_count = 0

        # --- locked-in after warmup ---
        self.fps_locked = None
        self.target_len = None
        self._window_n = None
        self._calib_n = None
        self._shapes = deque(maxlen=4096)   # recent normalised shapes

        # --- calibrate / detect state ---
        self._baseline = []
        self.threshold = None
        self.motion_state = False      # False = still, True = motion (hysteresis)
        self._last_motion = None

        # --- Hampel filter state (FIFO buffer of recent amp vectors) ---
        self._hampel_buf = []

        # --- Guard/DC mask state (computed once target_len is known) ---
        self._guard_mask = None

        # --- Component 2: subcarrier-selection state ---
        self._selection = None          # SubcarrierSelection once adapted
        self._sel_active = None         # bool array over bins (guard-free & selected)
        self._stills = []               # amplitude frames captured as still references
        self._motions = []              # amplitude frames captured during motion
        self._sel_pending_after = 0     # adapt once this many motion frames land
        self._sel_cooldown = 0          # capture-only counter between adaptations
        self._adapt_n = 60              # frames per class before re-selecting
        self._reselects = 0             # adaptations completed

    def _snapshot(self):
        calib = 0.0
        if self._calib_n:
            calib = min(1.0, len(self._baseline) / self._calib_n)
        return DetectorState(
            phase=self.phase,
            in_motion=self.motion_state,
            motion_level=self._last_motion,
            threshold=self.threshold,
            fps=self.fps_locked,
            target_len=self.target_len,
            calib_progress=calib,
        )

    def _apply_hampel(self, amp):
        """Hampel outlier filter along the time axis, per subcarrier.

        Keeps a short FIFO of recent frames; for each subcarrier, if the current
        value deviates from the local median by more than HAMPTHR * sigma_est
        (sigma_est = 1.4826 * MAD), it is replaced with that median.
        """
        arr = np.asarray(amp, dtype=float)
        n_sc = len(arr)
        res = arr.copy()
        self._hampel_buf.append(arr.copy())
        if len(self._hampel_buf) > HAMPHALF + 1:
            self._hampel_buf.pop(0)
        if len(self._hampel_buf) < HAMPHALF + 1:
            return res                      # too few samples yet: pass through
        for c in range(n_sc):
            vals = [row[c] for row in self._hampel_buf]
            cs = np.sort(vals)
            med = float(cs[len(cs) // 2])
            mad = float(np.median(np.abs(np.asarray(vals) - med)))
            sigma = 1.4826 * mad
            dev = abs(arr[c] - med)
            eps = 1e-15
            if (sigma > eps and dev > HAMPTHR * sigma) or (sigma <= eps and dev > eps):
                res[c] = med
        return res

    def update(self, amp, t=None) -> DetectorState:
        """Feed one amplitude vector; return the current DetectorState."""
        if t is None:
            t = time.time()
        self._fps_times.append(t)

        # --- WARMUP: measure fps + lock the dominant frame length ---
        if self.phase == "warmup":
            self._len_votes[len(amp)] += 1
            if self._measure_start is None:
                recent_1s = sum(1 for ts in self._fps_times if ts >= t - 1.0)
                if recent_1s >= self.warmup_settle_fps:   # stream is up: start measuring
                    self._measure_start, self._measure_count = t, 0
                return self._snapshot()
            self._measure_count += 1
            if t - self._measure_start >= self.warmup_measure_sec:
                # Median interval is immune to USB burst inflation.
                self.fps_locked = max(_median_interval_fps(self._fps_times), 1.0)
                self.target_len = self._len_votes.most_common(1)[0][0]
                self._window_n = max(8, round(self.window_sec * self.fps_locked))
                self._calib_n = max(20, round(self.calib_sec * self.fps_locked))
                self._guard_mask = combined_mask(self.target_len)
                self._shapes.clear()
                self._hampel_buf = []
                self.phase = "calibrate"
            return self._snapshot()

        # Ignore off-type frames so the window stays consistent.
        if len(amp) != self.target_len:
            return self._snapshot()

        # 1. Hampel filter: remove spike outliers before anything else.
        amp_clean = self._apply_hampel(amp)

        # 2. Guard/DC mask + component-2 subcarrier selection:
        #    onlybins surviving BOTH the static mask and the sensitivity
        #    selection contribute to any statistic.
        gm = self._guard_mask
        if gm is not None:
            base = ~gm
            amp_clean[gm] = amp_clean[base].min() * 0.5
        else:
            base = np.ones(len(amp_clean), dtype=bool)
        if self._sel_active is not None:
            active = base & self._sel_active
        else:
            active = base

        # 3. Gain removal over ACTIVE subcarriers only, so masked bins contribute nothing.
        shape = np.ones_like(amp_clean)
        shape[active] = amp_clean[active] / (amp_clean[active].mean() + 1e-6)
        self._shapes.append(shape)
        if len(self._shapes) < self._window_n:
            return self._snapshot()

        # 4. Motion level = mean per-subcarrier std over ACTIVE bins only.
        arr = np.array(self._shapes)                 # (window_n, target_len)
        sc_std = arr.std(axis=0)
        motion = float(sc_std[active].mean())
        self._last_motion = motion

        # --- Component 2: feed the still/motion reference buffers ---
        # amp_clean (not raw amp): guard/DC bins were just neutralised to a
        # constant, so their variance ratio collapses in _adapt and they can
        # never be re-selected. Full length is preserved so buffer column
        # indices always match target_len — stripping would misalign every
        # mask built from the selection.
        self._sel_feed(amp_clean)

        # --- CALIBRATE: collect the still baseline, then fix the threshold ---
        if self.phase == "calibrate":
            self._baseline.append(motion)
            if len(self._baseline) >= self._calib_n:
                self.threshold = max(
                    float(np.percentile(self._baseline, 95) * self.thresh_mult), 1e-4)
                self.phase = "detect"
            return self._snapshot()

        # --- DETECT: hysteresis state machine ---
        if self.motion_state and motion < self.threshold * self.hyst_low:
            self.motion_state = False
        elif (not self.motion_state) and motion > self.threshold:
            self.motion_state = True
        return self._snapshot()

    def _sel_feed(self, amp):
        """Component 2: collect still/motion references and adapt when ready.

        During "calibrate" (known-still) it fills the still buffer. In "detect"
        it tags frames as motion while the hysteresis machine is open. When both
        buffers hold _adapt_n frames it runs the variance-ratio selection and, if
        the refactored metric is a clear improvement, commits it (see _adapt).
        Buffers are capped, so between adaptations only the most recent frames
        are kept and the cost per frame stays O(1).
        """
        if self._sel_cooldown > 0:
            self._sel_cooldown -= 1
            return
        if self.motion_state:
            buf = self._motions
        elif self.phase == "calibrate":
            buf = self._stills
        elif not self._selection:
            # First adaptation needs a ride-along motion class after the
            # baseline window: open on churn above a wide margin of the
            # threshold (1.6x, not 1.0x, so hysteresis chatter does not feed
            # borderline noise in as 'motion').
            if self.threshold and self._last_motion and self._last_motion > self.threshold * 1.6:
                buf = self._motions
            else:
                return
        else:
            return
        # Full-length neutralised frames: indices stay aligned with target_len,
        # and the neutralised constant makes dead bins carry ~zero variance in
        # both classes, so their motion/static ratio is ~1 and never selected.
        buf.append(amp)
        if len(buf) > self._adapt_n * 2:
            del buf[: len(buf) - self._adapt_n * 2]
        if len(self._stills) >= self._adapt_n and len(self._motions) >= self._adapt_n:
            self._adapt()

    def _adapt(self):
        """Component 2: run the selection and commit it if clearly better.

        Deciding with the refactor ratio (new metric / old metric over the same
        motion frames) rather than on an absolute number keeps the decision
        scale-free for a threshold that can sit anywhere after calibrate.
        """
        still = np.array(self._stills)
        motion = np.array(self._motions)
        try:
            sel = select_sensitive_subcarriers(
                motion, still, top_k=SEL_TOP_K, min_sensitivity=SEL_MIN_SENSITIVITY)
        except (ValueError, IndexError):
            self._stills = []
            self._motions = []
            return
        if not sel.selected_indices:
            # No bin clears the floor: drop the buffers and retry later
            self._stills = []
            self._motions = []
            return
        new_active = np.zeros(self.target_len, dtype=bool)
        new_active[sel.selected_indices] = True
        # The committed selection must exclude guard/DC bins forever: intersect
        # with the guardfree base so a late-arriving ratio can never re-open one.
        if self._guard_mask is not None:
            new_active &= ~self._guard_mask
        if not new_active.any():
            self._stills = []
            self._motions = []
            return
        # Compare metrics computed on the SAME motion frames
        old_active = (self._sel_active if self._sel_active is not None
                      else ~self._guard_mask)
        motion_old = float(motion.std(axis=0)[old_active].mean())
        motion_new = float(motion.std(axis=0)[new_active].mean())
        if motion_new <= SEL_RATE * motion_old:
            # No measurable improvement: keep current bins, stop adapting
            self._stills = []
            self._motions = []
            self._sel_cooldown = 5 * self._calib_n
            return
        self._selection = sel
        self._sel_active = new_active
        self._reselects += 1
        # Re-read only the threshold, not the frames: the motion metric is now
        # computed on far fewer bins, so re-derive it from the same still data
        self._rebase(new_active)

    def _rebase(self, active):
        """Component 2: recompute the threshold under the narrowed metric.

        Rebuilds the still baseline live from the canonical still frames via a
        replay, so calibration never has to pause for the change.
        """
        shapes = []
        for amps in self._stills[: self._calib_n]:
            a = np.asarray(amps, dtype=float)
            shape = np.ones(a.shape[0])
            shape[active] = a[active] / (a[active].mean() + 1e-6)
            shapes.append(shape)
        stds = np.array(shapes).std(axis=0)
        self.threshold = max(float(np.percentile(stds, 95) * self.thresh_mult), 1e-4)
        self._stills = []
        self._motions = []

    def recalibrate(self):
        """Re-learn the still baseline (e.g. after moving the board).

        Keeps the already-measured fps and locked frame length; only the baseline
        and threshold are re-derived. Call this, then hold still for ~calib_sec.
        """
        if self.target_len is None:        # never finished warmup: nothing to keep
            return
        self._baseline = []
        self.threshold = None
        self.motion_state = False
        self._shapes.clear()
        self._hampel_buf = []
        self._guard_mask = None
        # Component 2 state resets with the baseline it was derived from
        self._stills = []
        self._motions = []
        self._selection = None
        self._sel_active = None
        self.phase = "calibrate"
