# RuView Integration ROI Report

**Subject:** Which RuView components are worth integrating into WiFi Space Mapper (WSM), ranked by return on engineering effort.
**Date:** 2026-06-30 (repo state: WSM `03e164b`, RuView `fefb0f91`)
**Status:** Research findings only. **No code was modified.** No recommendation in this document has been implemented.

---

## 0. Executive summary

WSM is a small, coherent, hardware-verified single-node CSI project (~2,300 lines total: 158 lines of
firmware, ~1,500 lines of host Python, 7 tools). RuView is a 100k+-line research monorepo spanning Rust,
TypeScript, Python, C firmware, WASM, and ~420 ADRs.

The honest headline: **the overlap between "what RuView has" and "what WSM can actually use on one
classic ESP32" is small, and it is concentrated in the DSP front end.** RuView's headline capabilities —
pose estimation, vital signs, multi-person counting, through-wall imaging — are either multi-node-gated,
trained-model-gated, or explicitly documented by RuView itself as unvalidated. Its real value to WSM is
three well-tested, dependency-light signal-processing algorithms that fix concrete, *reproducible* defects
I measured in WSM's current pipeline.

**Top three changes (detail in section 7):**

1. **Fix the frame-rate estimator, then add RuView's Hampel filter + guard/DC subcarrier masking.**
2. **Adopt RuView's variance-ratio subcarrier selection as an opt-in sensitivity-map diagnostic.**
3. **Add RuView's data-provenance/quality gating pattern (with banner) to kill silent blank views.**

Everything else is ranked and justified in section 5, including an explicit reject list (section 7.4).

---

## 1. Method and evidence standard

I mapped both repositories, then classified every claim by the *strongest* evidence available:

| Tier | Meaning |
|---|---|
| **V - Verified by me** | I compiled and/or executed the code in this session and observed the result. |
| **T - Tested upstream** | RuView ships passing unit tests; I read them and, where cheap, re-ran them. |
| **I - Implemented, unmeasured** | Code exists and is referenced, but I found no measurement of sensing accuracy. |
| **P - Prototype / heuristic** | Skeleton, mock, simulation, or a constant that is not a measurement. |
| **D - Documented only** | ADR/design text with no corresponding implementation. |

The distinction matters because RuView's README markets the whole system uniformly. It does not present
pose, vitals, and multi-person counting as having the same maturity as its DSP primitives, and neither
should we.

### 1.1 What I actually executed

- WSM `tools/selftest_csi.py` -> **exit 0**, 62 subcarriers, detector separates STILL/MOTION.
- WSM `tools/wifi_visualizer.py --demo --smoketest` under `QT_QPA_PLATFORM=offscreen` and `minimal` -> **exit 0**.
- WSM visualizer instrumented headlessly for 140-400 ticks -> pipeline produces real data (see section 4).
- RuView `hampel.rs` copied into a standalone crate (dependency: `thiserror` only) -> **10/10 tests pass**.
- RuView `subcarrier_selection.rs` variance-ratio half, standalone (`thiserror` + `ndarray`) -> **6/6 tests pass**.
- A reproduction of WSM's USB-burst timing against the live `MotionDetector` -> **found a real defect** (section 4.2).

---

## 2. Architecture, dependencies, licenses

### 2.1 WiFi Space Mapper

| Layer | Files | Notes |
|---|---|---|
| Firmware (C / ESP-IDF 5.5) | `main/main.ino` (158 ln) | STA connect -> CSI enable -> gateway self-ping @10 ms -> CSV over serial @921600 |
| Engine | `tools/csi_stream.py` (362 ln) | `SerialSource` / `ReplaySource` / `SyntheticSource` behind threaded `CSIStream` |
| Detector | `tools/detector.py` (173 ln) | gain removal -> moving-window std -> P95 x 1.4 threshold -> hysteresis |
| Visualizer | `tools/wifi_visualizer.py` (564 ln) | PyQtGraph + PySide6, 6 tabs |
| Simple viewer | `tools/live_csi_plot.py` (115 ln) | matplotlib |
| Recorder | `tools/record_csi.py` (184 ln) | labeled walk segments -> `.npz` |
| Utilities | `csi_face.py`, `make_previews.py`, `selftest_csi.py` | |

**Dependencies:** `numpy`, `pyserial`, `matplotlib`, `pyqtgraph`, `PySide6`. Pure host-side; nothing exotic.

**CSI format (WSM):** plain-text serial line, one per received frame:

```
CSI_DATA,<len>,<rssi>,[i0 q0 i1 q1 ...]
```

Interleaved signed bytes, `(imag, real)` pairs. Host drops the first 2 pairs (`SKIP_HEAD_PAIRS = 2`),
computes `|H| = sqrt(re^2 + im^2)`, and discards phase by design. Firmware captures **Legacy LTF only**,
giving uniform **64 subcarriers -> 62 after the head drop**.

**License: none.** There is no `LICENSE` file in WSM. This is a real gap (see risk R1).

### 2.2 RuView

Vast. Relevant subtrees only:

| Area | Path | Language |
|---|---|---|
| Signal DSP | `v2/crates/wifi-densepose-signal/src/` | Rust (~30 modules + 20 `ruvsense/` modules) |
| ESP32 firmware | `firmware/esp32-csi-node/main/` | C / ESP-IDF, ESP32-S3 and ESP32-C6 |
| Python bindings | `python/wifi_densepose/` | PyO3 (`_native`) + thin client |
| Web UI | `ui/`, `dashboard/` | JS/TS, Three.js, Lit |
| Host scripts | `scripts/` | Python + Node CLI tools |
| Design docs | `docs/adr/ADR-*.md` | ~420 files |

**License:** root `LICENSE` is **MIT** (c) 2024 rUv. The Rust workspace (`v2/Cargo.toml:164`) is
**MIT OR Apache-2.0**. Firmware is MIT OR Apache-2.0. Both are permissive and compatible with WSM's
(currently unstated) terms. The vendored `vendor/*` submodules carry their own licenses - check
per-submodule if you ever pull from there.

**CSI format (RuView):** binary ADR-018 over **UDP**, not serial:

```
offset 0  u32 LE  magic 0xC5110001
offset 4  u8      node_id
offset 5  u8      n_antennas
offset 6  u16 LE  n_subcarriers
offset 8  u32 LE  freq_mhz
offset 12 u32 LE  sequence
offset 16 i8      rssi
offset 17 i8      noise_floor
offset 18 u8      ppdu_type
offset 19 u8      flags
offset 20 ...     n_antennas * n_subcarriers int8 I/Q pairs
```

Reference decoders: `scripts/record-csi-udp.py:parse_csi_packet` and `scripts/collect-training-data.py`
(both agree on the `<IBBHIIbbBB` 20-byte header).

**This is a wire-format mismatch with WSM**, but it is not a blocker: the *algorithms* are format-agnostic
and the *format* is a small adapter.

---

## 3. Format compatibility - the key question

WSM emits text CSV over serial; RuView emits binary over UDP. Nothing in RuView consumes WSM's line format
today.

**But the incompatibility is shallow.** Concretely:

- WSM's `amp` vector (62 floats/frame) and RuView's amplitude array (64 floats/frame on ESP32-S3) are the
  same kind of object: per-subcarrier `|H|` on a 20 MHz channel.
- RuView's reusable DSP modules operate on `&[f64]` / `Array2<f64>` slices. They do not know or care where
  the numbers came from.
- The frame-count difference (62 vs 64) is not a problem for the modules I recommend, because WSM's
  `SKIP_HEAD_PAIRS = 2` is a display convenience, not a physical tone map. RuView's
  `HardwareNormalizer::detect_hardware` keys off counts like 56/64/114/242; 62 would land in the `Generic`
  bucket. That also means RuView's fixed tone maps (`HT20` 52-active, `he20` 256-bin) do **not** apply
  cleanly to WSM's post-drop vector.

**Verdict:** adapt modules; do not adopt RuView's wire format. Changing WSM's serial path would violate the
brief's constraint and buys nothing - the algorithms port to Python in a few dozen lines each.

---

## 4. The blank visualization issue

The brief asked whether RuView offers a fix or a reusable plotting component. I investigated directly.

### 4.1 What I found

**Good news: the pipeline is not broken.** Instrumenting the visualizer headlessly produced:

```
C = 62
latest amp len = 62
wf shape = (62, 256)   level_hi = 30.68
det phase = calibrate, threshold = None, motion_level = None
motion_hist len = 0
stream.fps = 80.9, total frames = 358
```

Frames parse, the waterfall fills, amplitudes are sane (0.36 ... 33.1). Rendering each tab to PNG and
measuring non-background pixel coverage:

| Tab | non-background | bright | unique colours |
|---|---|---|---|
| Dashboard | 18.2 % | 10.5 % | 3086 |
| Raw CSI | 4.7 % | 2.0 % | 4149 |
| Spectrogram | 63.2 % | 36.2 % | 1687 |
| **Motion** | **2.8 %** | **1.4 %** | 528 |
| **Radar** | **2.7 %** | **1.1 %** | 605 |
| **Doppler** | **3.3 %** | **1.5 %** | 744 |

Motion, Radar, and Doppler are ~27-33x emptier than the Spectrogram. **That is the blank visualization.**
It is not a crash and not a data problem - it is `motion_hist` being empty for a long time.

### 4.2 Root cause: the fps estimator is corrupted by USB bursting

WSM's own `CHANGELOG.md` and `detector.py` docstring (lines 20-23) document that USB-serial delivers
frames in ~16 ms bursts and that "any short estimate lies." The mitigation is a 3-second averaging window
(`WARMUP_MEASURE_SEC = 3.0`).

**That mitigation does not work**, because the averaging window is itself inside the burst artifact. I
reproduced live-like timing (frames delivered ~2 per 16 ms, i.e. 125 fps on the wire) against the real
`MotionDetector`:

```
fps_locked = 125.0          # the true synthetic rate was 90
window_n   = 250            # intended ~180 at 90 fps  -> 39% too long
calib_n    = 1000           # intended ~720           -> 39% too long
first motion_level  at  5.1 s
first detect        at 13.1 s   <- the MOTION / STILL answer only appears here
```

So `motion_hist` (and therefore the Motion curve, the threshold line, and the Doppler FFT, which
`DopplerView.update_view` gates on `len(hist) < 64`) stays empty for **~5 seconds**, and the detector
cannot answer anything at all for **~13 seconds**. Meanwhile the telemetry bar happily shows `fps: 125` -
a number that is wrong, and wrong in the direction that lengthens every window.

The bug is a feedback loop: `window_n = round(window_sec * fps_locked)` uses a *fps* inflated by the burst
delivery pattern, so time constants stretch instead of staying rate-independent. The "rate-independent by
design" claim in the README is only true when frame arrivals are uniform.

### 4.3 Does RuView offer a fix?

**Partially - conceptually yes; as a drop-in plotting component, no.**

RuView's plotting is Three.js/WebGL (`ui/components/signal-viz.js` builds a `THREE.Group` of
`PlaneGeometry` cells), which is architecturally incompatible with PyQtGraph. Its dashboard
(`dashboard/src/`) is Lit/TS. **Reject the plotting components.**

What *is* directly valuable is RuView's discipline about honest data state, in two places:

1. `ui/utils/data-source-banner.js` - a viewport-fixed banner that *shows nothing when data is live* and
   appears only when the reading is untrustworthy ("RECONNECTING - the display is frozen at the last live
   reading"; "SIMULATED - every value on this screen is invented by your browser, not measured"). The
   source comment is the actual lesson: *"The per-tab banner was correct and still invisible where it
   mattered... a screen full of client-invented sensing frames was indistinguishable from measurement."*

2. `ADR-356` - "Rate truthful CSI and reliability gated room adaptation" - addresses precisely WSM's class
   of bug. Its decision list includes: estimate delivered rate from accepted monotonic timestamps; cap the
   physical input rate at the firmware contract; **reconfigure downstream filters only after a 20 % rate
   change**; and, most relevant, *"Export zero sample rate until the estimate is mature. Never present a
   startup fallback as a measured node property."* WSM currently presents a *measured-looking* 125 fps that
   is an artifact.

ADR-356 also documents a related hardware gotcha WSM will hit: `wifi_csi_info_t.first_word_invalid` marks
the first four CSI bytes invalid. RuView zeroes them in firmware
(`firmware/esp32-csi-node/main/csi_collector.c:269-278`) and sets a sanitation flag. WSM instead drops two
*pairs* on the host (`SKIP_HEAD_PAIRS = 2` = 4 bytes) - functionally similar, but blind: it does not check
`first_word_invalid`, so it discards four good bytes when they were valid and never signals when sanitation
occurred. **RuView's approach is the better reference here.**

---

## 5. Ranked opportunities

Effort is engineering time to a *validated* result, not to a first draft. "Evidence" uses the tiers from section 1.

| # | Component | Expected benefit | Evidence it works | Compat. with current ESP32 + format | Effort | Hardware / deps | Risks and limitations | Action |
|---|---|---|---|---|---|---|---|---|
| **1** | **Frame-rate estimator fix + Hampel filter + guard/DC masking** - `hampel.rs` (Hampel + MAD), `ADR-356` rate discipline | Removes the measured 13 s to first answer and the ~39 % window stretch; suppresses spike false-positives. Directly fixes the blank Motion/Doppler views. | **V**: `hampel.rs` compiles standalone (`thiserror` only), **10/10 tests pass**. **V**: I reproduced the 125 fps / 13.1 s defect on WSM's own detector. | Perfect - pure `&[f64]` math, format-agnostic, host-side, no firmware change | **Low** | None (port ~80 lines to NumPy; or `thiserror` in Rust) | Hampel adds latency if the window is large - keep `half_window=3` (upstream default). Does not by itself fix *calibration drift*. | **Reuse directly** (algorithm) |
| **2** | **Subcarrier sensitivity selection** - `subcarrier_selection.rs::select_sensitive_subcarriers` (variance ratio motion/static) + `select_by_variance` | WSM's roadmap already lists "subcarrier selection (drop guard/DC)". Upstream cites **6-10 dB SNR** gain (WiDance/WiGest). Turns 62 mixed bins into a ranked, defensible subset. | **V**: variance-ratio half compiles standalone, **6/6 tests pass**. **I** for the 6-10 dB figure - that is a literature claim, not a RuView measurement. | Good - pure matrix math; only needs a still period and a motion period, both of which WSM's calibrate phase already provides | **Low-Med** | `ndarray` (Rust) or NumPy | The 6-10 dB number is **not verified on WSM hardware**. `mincut_subcarrier_partition` needs `ruvector-mincut` - a heavy dep with no standalone test value; skip the min-cut half. | **Adapt** (variance-ratio only) |
| **3** | **Provenance / quality gating + "why is this blank" banner** - `data-source-banner.js` pattern, `esp32_jsonl_to_rvcsi.py::quality_and_status`, `ADR-346` fail-closed | Ends silent blank views; surfaces a stale/absent stream instead of an empty plot. The quality score also gives a principled frame-drop policy. | **I**: `quality_and_status` is a readable port of upstream validation logic; `data-source-banner.js` is plain JS. `ADR-346` is "Accepted and implemented" with a physical validation record (`docs/validation/2026-08-31-esp32-c6-occupancy-integrity.md`). | Host-side only | **Low** | None | RuView's specific thresholds (RSSI -110..0, quality >= 0.25) are tuned for its pipeline; re-derive or start permissive. The banner is a UX pattern, not code to copy verbatim. | **Adapt** (pattern + port) |
| **4** | **`first_word_invalid` handling + sanitation flag** | Removes a documented hardware artifact at the source; makes the data path honest about when bytes were rewritten. | **V/T**: implemented at `csi_collector.c:269-278`; `ADR-356` section "Firmware ingestion". | Requires a firmware change (touch `main.ino`) - the brief says preserve the serial path unless evidence proves otherwise. This is a narrow, evidence-backed exception; the host-side equivalent (check a flag instead of blindly dropping two pairs) needs no firmware change. | **Low** | None | Violates "prefer host-side" only if done in firmware. Do the **host-side** version first. | **Adapt** (host first) |
| **5** | **Delay-domain CIR view** - `ruvsense/cir.rs` (`CirEstimator`) | A genuinely new visualization (multipath delay profile) and the substrate for any future ranging. | **T**: 16 in-module tests plus 3 integration test files (`cir_synthetic.rs`, `cir_pipeline.rs`, `cir_ghost_taps.rs`). | `cir.rs` needs a proper tone map. WSM's 62-bin post-drop vector is not a clean HT20 grid, so the transform is approximate. Single-link CIR gives *range*, not position. | **Med** | None | WSM discards phase by design; CIR from amplitude alone is weak. `ADR-134` status is *Proposed*. | **Reference** |
| **6** | **Baseline/deviation calibration** - `ruvsense/calibration.rs` (`BaselineCalibration`, `PhyTier`, Welford + circular stats) | WSM's stated top weakness is "detection degrades if the environment or link changes after calibration." A stored, versioned empty-room baseline with a mismatch guard is the principled fix. | **T**: `ADR-135` is *Proposed*; the module has 10 in-file tests **plus** 3 integration files (`calibration_drift.rs`, `calibration_roundtrip.rs`, `calibration_synthetic.rs`). | `PhyTier` assumes 52/114/242/484 active tones; WSM's 62 does not map. It also requires phase, which WSM drops. | **Med-High** | None | The tier-mismatch guard would fire on WSM's vector. Phase-based deviation is unusable without phase; amplitude-only deviation is the usable subset. | **Reference** (amplitude subset) |
| **7** | **Coherence gate / quality-driven state machine** - `ruvsense/coherence_gate.rs` | "Accept / predict-only / reject / recalibrate" is a better detector state machine than WSM's binary hysteresis, and it *self-triggers recalibration* - exactly WSM's drift problem. | **T**: 18 in-file tests; `ADR-136`/`ADR-137` define the contracts. | Designed around multi-node coherence and Kalman tracking. A single node has no coherence *across* nodes, so only the temporal-drift subset applies. | **Med** | None | Applying it without a tracker means reimplementing the decision logic, not reusing it. Risk of over-engineering a 3-state problem. | **Reference** |
| **8** | **Recording / dataset format** - `.rvcsi` JSONL, `esp32_jsonl_to_rvcsi.py`, `collect-training-data.py` manifests | WSM's `record_csi.py` already does labeled `.npz` takes. RuView's JSONL adds self-describing headers, a validation policy, and a manifest for multi-session datasets. | **T**: `scripts/tests/` covers `test_calibration.py`, `test_capture_adr018.py` and others; the converter is complete and runs. | RuView's recorder listens on **UDP**; WSM is serial. The *schema* transfers; the transport does not. | **Med** | None | WSM's `.npz` plus embedded `meta` (person/session/take/fps/port) is already self-describing and lighter. Switching formats is churn unless multi-node capture is planned. | **Reference** |
| **9** | **Supervised training pipeline** - `scripts/train-*.py/js`, `wifi-densepose-train`, `cog-person-count` | WSM's Rung 2 (activity discrimination: empty/still/walking) is a classification task. RuView has real training scaffolding. | **I/P**: harnesses exist and CI runs them, but **no held-out accuracy number** is claimed for a single classic ESP32. `ADR-346` item 7 states the count "remains a heuristic until a leakage free, held out physical dataset demonstrates otherwise." | RuView models expect 56-bin canonical or 64-bin S3 input, with multi-antenna or multi-node context. | **High** | PyTorch/Candle, GPU, large labeled corpus | No pretrained model matches WSM's 62-bin single-antenna input. WSM has **zero** labeled training data (a recorder exists; no captures are committed). | **Reject for now** |
| **10** | **Pose estimation / DensePose (17 keypoints)** | RuView's headline feature. | **P**: RuView's own `ADR-029` section 1.1 states single-ESP32 torso jitter is **~15 cm RMS** and multi-person "fails >2 people, frequent ID swaps" - against a product requirement of <3 cm. Paths depending on it use synthetic or mock CSI (`ui/pose-fusion/js/csi-simulator.js`, `firmware/esp32-csi-node/main/mock_csi.c`). | Incompatible | **Very High** | Trained model, more nodes | **Physically unsupported on one 20 MHz single-antenna link** - no angle-of-arrival, ~7.5 m range resolution. WSM's README already states this correctly. | **Reject** |
| **11** | **Vital signs (breathing / heart rate)** | Attractive; RuView performs on-device bandpass + zero-crossing BPM. | **P/I**: the firmware README labels both **"Experimental"** and states *"not a medical measurement"*. `examples/research-sota/07-negative-results/README.md` documents a **negative result**: contactless BP is "5 dB short" and "provably worse than a $20 arm cuff." | Needs stable phase (WSM drops it) and >= 20 Hz clean timing. | **High** | None | RuView itself publishes this as a *negative* result. Presenting vitals would be exactly the "unvalidated vital-sign estimate" the brief forbids. | **Reject** |
| **12** | **Full RuView monorepo / dashboard / mesh** | n/a | n/a | Incompatible | **Very High** | Large | Explicitly out of scope by the brief; would duplicate subsystems WSM already has. | **Reject** |

---

## 6. Risks and blockers

**R1 - WSM has no LICENSE file.** RuView is MIT (root) and MIT-OR-Apache-2.0 (Rust). Copying code into an
unlicensed repo leaves the combined work's terms undefined. Adding a `LICENSE` to WSM (and a `NOTICE` or
attribution line for ported algorithms) is a prerequisite for items 1-3, and takes minutes. This is the
single cheapest risk item in this report.

**R2 - Dependency gravity.** The `wifi-densepose-signal` crate pulls `ruvector-mincut`,
`ruvector-attn-mincut`, `ruvector-attention`, `ruvector-solver`, `midstreamer-temporal-compare`,
`midstreamer-attractor`, and, via the default `eigenvalue` feature, `ndarray-linalg` -> **BLAS/LAPACK**.
Taking the crate wholesale would drag a large, partly git-sourced dependency tree into a five-file Python
project. **Port the algorithms; do not vendor the crate.** My standalone tests prove the two recommended
modules are self-contained (item 1 needs only `thiserror`; item 2 needs `thiserror` + `ndarray`).

**R3 - Hardware mismatch is real but narrow.** RuView targets **ESP32-S3 / ESP32-C6**; WSM runs a classic
**ESP32-WROOM-32**. RuView's `ADR-028` capability audit lists classic ESP32 at **56 subcarriers** and the S3
at 64. WSM reports 64 -> 62 post-drop. Consequences: RuView's fixed tone maps (`PhyTier::ht20` etc.) do not
apply, and its `EDGE_MAX_SUBCARRIERS` (128/256) is irrelevant here. This blocks items 5 and 6 and any model
reuse - which is precisely why items 1-3 were chosen: they are tone-map-free.

**R4 - RuView's 50 Hz firmware cap vs WSM's 87 fps.** WSM deliberately pushes to ~87 fps, and its
`CHANGELOG.md` notes the next wall is the printf-in-Wi-Fi-task. RuView *gates callbacks down to 50 Hz*
(`CSI_MIN_PROCESS_INTERVAL_US`) to avoid a documented LoadProhibited panic in `cache_ll_l1_resume_icache`
from WiFi-ISR / SPI flash-cache races. **This is a genuine warning**: WSM is running 74 % above the rate at
which RuView found the ESP32 to be unstable, and WSM's own comment acknowledges the printf cost. WSM has
been stable in practice, but this deserves a long-run soak test.

**R5 - Evidence quality varies wildly across RuView.** Some modules are genuinely well-tested (`hampel.rs`,
`subcarrier_selection.rs`, `multistatic.rs` with 47 tests). Others are scaffolds. Many ADRs are *Proposed*.
The README's badges do not distinguish these. Assume nothing from the README; the tiers in section 1 are
the filter.

**R6 - No CI or test target in WSM.** `tools/selftest_csi.py` is the only automated check, and there is no
CI. Any new DSP stage should extend that self-test, or regressions will be invisible.

---

## 7. Roadmap

### 7.1 Quick wins - minimal changes, immediately useful

**QW1. Fix the frame-rate estimator** (no RuView code needed; RuView's ADR-356 is the design reference).

- Use the **median** inter-frame arrival interval (robust to the ~16 ms burst quantization) instead of
  frame-count / elapsed.
- Track a long-run rate and only re-tune windows on a >20 % change, per `ADR-356`.
- Export `None` / `0` until the estimate is mature, rather than publishing a fallback as measured.
- Expected: `fps_locked` approaches the true rate; window and calibration stop stretching; the first answer
  returns to ~8 s.

**QW2. Hampel filter on the amplitude vector before gain removal.**

- Port `v2/crates/wifi-densepose-signal/src/hampel.rs` (~80 lines; `half_window=3`, `threshold=3.0`,
  `MAD_SCALE=1.4826`, `ZERO_MAD_EPSILON=1e-15`) to NumPy.
- Expected: spikes stop masquerading as motion; the upstream tests port directly as assertions.

**QW3. Guard/DC subcarrier masking plus honest blank states.**

- Mask the guard/DC bins WSM's README already identifies (band edges plus the DC spike) instead of letting
  near-zero-variance bins dilute the mean of the per-subcarrier std.
- Add a state badge distinguishing `NO DATA` / `STALE` / `CALIBRATING` / `LIVE` (the
  `data-source-banner.js` pattern), so a blank Motion tab always says *why*.
- Expected: the 2.8 %-coverage blank tabs become self-explanatory; small SNR gain.

### 7.2 Next-stage improvements - modest integration work

**NS1. Variance-ratio subcarrier selection.** Port `select_sensitive_subcarriers` and `select_by_variance`
from `subcarrier_selection.rs` (skip the `ruvector-mincut` half). WSM's calibrate phase already provides the
still period; a "wave your hand" prompt provides the motion period. Surface the result as a **Sensitivity**
tab (bar chart of scores) and optionally feed the top-K into the detector. Measure the SNR claim on WSM
hardware rather than trusting WiDance's 6-10 dB.

**NS2. Quality-scored frame admission.** Port the spirit of
`esp32_jsonl_to_rvcsi.py::quality_and_status` (dead-subcarrier fraction, amplitude spike vs median, RSSI
bounds) into `csi_stream.py` as a per-frame quality score with a degrade-not-reject policy. Feed quality
into the detector so a bad stretch suppresses `in_motion` rather than producing a false MOTION.

**NS3. Baseline snapshot with validity guard.** Use the *amplitude-only* subset of
`ruvsense/calibration.rs`: store per-subcarrier mean/std from a known-empty capture, persist it next to the
recording, and detect when live statistics drift beyond a threshold, then prompt "environment changed,
recalibrate." This directly targets WSM's documented weakness. Drop the `PhyTier` and phase machinery.

### 7.3 Research projects - uncertain payoff, substantial effort

**RP1. Amplitude-only delay-domain (CIR-like) view.** Reference `ruvsense/cir.rs`. Requires deciding how to
treat WSM's non-standard 62-bin vector. Might yield a compelling new visual; may be physically
uninformative without phase.

**RP2. Activity classification (WSM Rung 2).** Feasible in principle, but WSM has **no labeled data** and
RuView has no transferable pretrained model for 62-bin single-antenna input. Requires a real recording
campaign first (`tools/record_csi.py` already exists for it), then a small from-scratch model.

**RP3. Multi-node.** Explicitly gated on hardware WSM does not have. RuView's own `ADR-029` section 1.1 is
the evidence that one node is insufficient for localization. Do not start here.

### 7.4 Avoid for now

- **Pose estimation / DensePose** - RuView's own ADR states ~15 cm jitter and multi-person failure on a
  single ESP32; several paths run on mock or simulated CSI.
- **Vital signs** - labelled "Experimental" by RuView, "not a medical measurement" - and RuView publishes a
  *negative* result for the adjacent BP use case.
- **Vendoring the `wifi-densepose-signal` crate** - dependency gravity (R2).
- **`ruvector-mincut`-based `mincut_subcarrier_partition`** - heavy dependency, extra complexity, no
  standalone test value; variance-ratio already answers the question.
- **RuView's UI/plotting stack** (Three.js `signal-viz.js`, Lit `dashboard/`) - architectural mismatch with
  PyQtGraph.
- **`.rvcsi` format migration** - WSM's `.npz` plus `meta` is already adequate; churn without payoff.
- **Multi-person counting** - `ADR-346` section 7 and item 7 state plainly that it "remains a heuristic
  until a leakage free, held out physical dataset demonstrates otherwise."

---

## 8. Final recommendation - top three changes

Implement **in this order**. Each is independently valuable, so stopping after any step leaves the project
better off than it started.

### 1st - Fix the rate estimator and the detector's time constants

**Why first:** it is the root cause of the blank-visualization complaint, it needs **no RuView code**, and it
invalidates the "rate-independent" claim the rest of the roadmap rests on. Everything after this is measured
against a correct clock.

- **Expected outcome:** first `MOTION`/`STILL` answer returns from ~13 s to ~8-10 s; `window_n` and
  `calib_n` match their intended second-based values; the Motion and Doppler tabs populate ~2.5x sooner; the
  telemetry fps stops reporting a fabricated 125.
- **Validation:** extend `tools/selftest_csi.py` with a **burst-timed** synthetic case (frames emitted ~2 per
  16 ms) asserting `fps_locked` is within 10 % of truth and `window_n` is approximately `2.0 * fps`. Confirm
  on hardware that the Motion tab shows a threshold line within ~10 s of launch.

### 2nd - Hampel filter, guard/DC masking, and honest blank states

**Why second:** highest accuracy-per-line-of-code, and it makes failures *visible* instead of blank - which is
what made step 1 hard to diagnose in the first place.

- **Expected outcome:** spike-induced false positives removed; guard/DC bins no longer dilute the motion
  metric; an empty tab always explains itself (`NO DATA` vs `CALIBRATING` vs `STALE`).
- **Validation:** port `hampel.rs`'s 10 tests as Python assertions (clean signal unchanged; single spike
  replaced by median; 30 % contamination still detected). Re-run `selftest_csi.py` and confirm zero false
  positives over a still-room hardware run - WSM's README claims zero today, so this must not regress. Add a
  deliberate "unplug the board" check that the UI reports `NO DATA`.

### 3rd - Variance-ratio subcarrier selection

**Why third:** it is the one place RuView offers a *capability* WSM has already committed to building
("subcarrier selection" in the README roadmap), with verified working code - but its benefit is a literature
claim, so it must be measured, not assumed.

- **Expected outcome:** a ranked sensitivity map over the subcarriers, plus an optional top-K detector input.
  If the SNR gain is real on this hardware, it should show up as a wider margin between the still-room level
  and the threshold.
- **Validation:** port `subcarrier_selection.rs`'s 6 tests as Python assertions. Then measure, on hardware:
  still-room motion level and threshold with all bins vs top-K bins, over >= 3 runs each, and record the
  *ratio* (level / threshold). **Report the number whether or not it flatters the idea** - if the ratio does
  not improve, keep the sensitivity view as a diagnostic and drop the top-K filter.

### Prerequisite (do before step 1)

Add a `LICENSE` to WSM. RuView is MIT / MIT-OR-Apache-2.0 and both are compatible, but an unlicensed target
repo makes the combined terms undefined (R1). Add an attribution note listing ported algorithms and their
source files.

---

## Appendix A - Reference map (paths verified in this session)

**WiFi Space Mapper**

```
main/main.ino:61                  CSI CSV printf (the wire format)
main/main.ino:130-143             wifi_csi_config_t (Legacy LTF only -> 64 subcarriers)
tools/csi_stream.py:39            PATTERN (the CSV regex)
tools/csi_stream.py:44            SKIP_HEAD_PAIRS = 2
tools/csi_stream.py:47-61         decode_tokens (imag/real -> |H|)
tools/detector.py:40-45           WARMUP_SETTLE_FPS / WARMUP_MEASURE_SEC / WINDOW_SEC / CALIB_SEC
tools/detector.py:106-128         warmup: fps measurement + frame-length lock   <- defect site
tools/detector.py:135-142         gain removal + moving-window std
tools/wifi_visualizer.py:356-372  DopplerView.update_view   <- the len(hist) < 64 blank gate
tools/wifi_visualizer.py:477-491  _ingest (waterfall + motion_hist)
tools/selftest_csi.py             the only automated test
docs/VISUALIZER.md                user guide; documents the 3-phase detector
CHANGELOG.md                      hardware-verified build history
```

**RuView**

```
LICENSE                                                            MIT (c) 2024 rUv
v2/Cargo.toml:164                                                  license = "MIT OR Apache-2.0"
v2/crates/wifi-densepose-signal/Cargo.toml                         heavy deps (see risk R2)
v2/crates/wifi-densepose-signal/src/hampel.rs                      <- RECOMMENDED (verified 10/10)
v2/crates/wifi-densepose-signal/src/subcarrier_selection.rs        <- RECOMMENDED (verified 6/6)
v2/crates/wifi-densepose-signal/src/motion.rs                      18 tests
v2/crates/wifi-densepose-signal/src/ruvsense/calibration.rs        ADR-135, 10 tests + 3 integ files
v2/crates/wifi-densepose-signal/src/ruvsense/cir.rs                ADR-134, 16 tests + 3 integ files
v2/crates/wifi-densepose-signal/src/ruvsense/coherence_gate.rs     18 tests
firmware/esp32-csi-node/main/csi_collector.c:154-281               ADR-018 serializer
firmware/esp32-csi-node/main/csi_collector.c:269-278               first_word_invalid sanitation
firmware/esp32-csi-node/main/edge_processing.c:1199-1200           on-device calibration
scripts/record-csi-udp.py:parse_csi_packet                         ADR-018 decoder (Python)
scripts/esp32_jsonl_to_rvcsi.py:quality_and_status                 validation/quality policy
scripts/collect-training-data.py                                   dataset labeling
ui/utils/data-source-banner.js                                     <- provenance UX pattern
docs/adr/ADR-028-esp32-capability-audit.md                         ESP32 vs S3 vs C6 subcarrier counts
docs/adr/ADR-029-ruvsense-multistatic-sensing-mode.md section 1.1  single-ESP32 limits (15 cm jitter)
docs/adr/ADR-135-empty-room-baseline-calibration.md                Status: Proposed
docs/adr/ADR-346-fail-closed-edge-occupancy-evidence.md            count is a heuristic (item 7)
docs/adr/ADR-356-rate-truthful-csi-and-reliability-gated-adaptation.md   <- directly relevant to QW1
examples/research-sota/07-negative-results/README.md               contactless BP negative result
docs/validation/2026-08-31-esp32-c6-occupancy-integrity.md         physical validation record
```

## Appendix B - Reproduction steps for the measured defect

```python
# WSM's detector, fed live-like USB burst timing (2 frames per 16 ms).
from detector import MotionDetector
from csi_stream import SyntheticSource

src = SyntheticSource(fps=90)
det = MotionDetector()
t, i = 0.0, 0
while t < 40.0:
    t += 0.016                 # one USB burst boundary
    for _ in range(2):         # 2 frames land per burst
        amp, _, _, _ = src.sample(i / 90.0); i += 1
        st = det.update(amp, t=t)

print(det.fps_locked)          # -> 125.0  (truth: 90)
print(det._window_n)           # -> 250    (intended: ~180)
```

Observed: `fps_locked = 124.9997`, `window_n = 250`, first `detect` at 13.12 s (vs ~8 s intended).

---

*Report complete. No files were modified. The recommendations above are unexecuted pending explicit instruction.*
