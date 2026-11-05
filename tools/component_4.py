"""Subcarrier quality, guard masking, and frame quality scoring for WiFi Space Mapper.

Port from RuView calibration framework.

Features:
  1. GUARD_MASK / DC_MASK -- Boolean masks marking known-irrelevant subcarriers
     so they are excluded from motion metrics instead of diluting them with noise.

  2. Frame quality score -- evaluates dead bins, amplitude spikes, RSSI range,
     and returns a 0..1 quality score (degrade-instead-of-reject policy).

Usage::
    gmask = make_guard_mask(n_subcarriers=62, drop_head_pairs=2)
    dmask = dc_mask(n_subcarriers=62)
    q, status, reasons = quality_score(amp, rssi)
"""

import numpy as np
from collections import Counter


MIN_SUBCARRIERS = 4
MAX_SUBCARRIERS = 4096
RSSI_LO, RSSI_HI = -110, 0
MIN_QUALITY = 0.25


def make_guard_mask(n_subcarriers=62, drop_head_pairs=2):
    """Create boolean mask marking guard band subcarriers."""
    mask = np.zeros(n_subcarriers, dtype=bool)
    lower_guards = min(drop_head_pairs, n_subcarriers // 4)
    mask[:lower_guards] = True
    upper_start = max(0, n_subcarriers - 8)
    mask[upper_start:] = True
    return mask


def dc_mask(n_subcarriers=62):
    """Boolean mask marking the DC subcarrier (index ~n//2)."""
    mask = np.zeros(n_subcarriers, dtype=bool)
    idx = n_subcarriers // 2
    for delta in (-1, 0, 1):
        pos = idx + delta
        if 0 <= pos < n_subcarriers:
            mask[pos] = True
    return mask


def combined_mask(n_subcarriers=62, drop_head_pairs=2):
    """Combined guard + DC mask."""
    return make_guard_mask(n_subcarriers, drop_head_pairs) | dc_mask(n_subcarriers)


def apply_mask(amp, mask, fill_value=np.nan):
    """Apply binary mask to amplitude vector."""
    result = amp.copy()
    result[mask] = fill_value
    return result


def exclude_masked_std(shapes, mask):
    """Compute per-subcarrier std excluding masked bins."""
    std_all = np.std(shapes, axis=0)
    std_clean = std_all.copy().astype(float)
    std_clean[mask] = np.nan
    return std_clean


def motion_level_with_mask(std_filtered, weight_by_sensitivity=True):
    """Compute mean motion level from filtered std."""
    valid = ~np.isnan(std_filtered)
    if valid.sum() == 0:
        return 0.0
    vals = std_filtered[valid]
    return float(vals.mean())


def quality_score(amp, rssi=None, spike_factor=50.0):
    """Score CSI frame quality. Ported from RuView validation logic."""
    amp = np.asarray(amp, dtype=float)
    sc = len(amp)
    reasons = []
    q = 1.0

    # Dead subcarriers
    if sc > 0:
        dead = int(np.sum(amp < 1e-6))
        frac = dead / sc
        if frac > 0:
            q *= max(1.0 - frac, 0.05)
            reasons.append(str(dead) + "/" + str(sc) + " dead subcarriers")

    # Amplitude spike check
    if sc >= 3:
        sorted_amp = np.sort(amp)
        median = max(sorted_amp[sc // 2], 1e-9)
        mx = float(sorted_amp[-1])
        if mx > median * spike_factor:
            q *= 0.7
            reasons.append("spike max=" + str(round(mx,1)) + " vs med=" + str(round(median,3)))

    # RSSI bounds
    if rssi is not None:
        if rssi < RSSI_LO or rssi > RSSI_HI:
            q *= 0.6
            reasons.append("rssi " + str(rssi) + " outside [" + str(RSSI_LO) + "," + str(RSSI_HI) + "]")
    else:
        q *= 0.95
        reasons.append("missing rssi")

    # Subcarrier count sanity
    if sc < MIN_SUBCARRIERS:
        q *= 0.5
        reasons.append("few SC: " + str(sc))
    elif sc > MAX_SUBCARRIERS:
        q *= 0.5
        reasons.append("many SC: " + str(sc))

    q = max(min(q, 1.0), 0.0)
    status = "Degraded" if q < MIN_QUALITY else "Accepted"
    return round(q, 3), status, reasons


def is_high_quality(amp, rssi=None):
    """Quick check: is this frame high-quality?"""
    q, _, _ = quality_score(amp, rssi)
    return q >= 0.8


def average_quality_for_buffer(amps_history, rssis=None):
    """Running average quality over recent frames."""
    if not amps_history:
        return 1.0, "Empty"
    n = min(len(amps_history), 100)
    total_q = 0.0
    statuses = []
    start = max(0, len(amps_history) - n)
    for i in range(start):
        amp = amps_history[i]
        rssi = rssis[i] if rssis and i < len(rssis) else None
        q, status, _ = quality_score(amp, rssi)
        total_q += q
        statuses.append(status)
    avg_q = total_q / n
    status_mode = Counter(statuses).most_common(1)[0][0] if statuses else "Unknown"
    return round(avg_q, 3), status_mode


if __name__ == "__main__":
    """Smoke test."""
    rng = np.random.default_rng(42)

    # Test guard mask
    gmask = make_guard_mask(62)
    assert gmask[:2].all(), "First 2 should be guarded"
    assert gmask[58:].all(), "Last 4 should be guarded"
    print("Guard: " + str(gmask.sum()) + "/62 flagged", flush=True)

    # Test DC mask
    dmask = dc_mask(62)
    print("DC: " + str(dmask.sum()) + "/62 flagged", flush=True)

    # Test combined
    cmask = combined_mask(62)
    print("Combo: " + str(cmask.sum()) + "/62 flagged", flush=True)

    # Test quality - good frame
    good = rng.normal(1.0, 0.1, 62)
    q, st, _ = quality_score(good, rssi=-60)
    print("Good: q=" + str(q) + " [" + st + "]", flush=True)
    assert q >= 0.8, "Good frame must have high quality: " + str(q)

    # Test quality - bad frame
    bad = rng.normal(1.0, 0.1, 62)
    bad[31] = 100.0
    bad[:10] = 0.0
    q2, st2, _ = quality_score(bad, rssi=-120)
    print("Bad: q=" + str(q2) + " [" + st2 + "]", flush=True)
    assert q2 < 0.8, "Bad frame must have low quality: " + str(q2)

    # Test masked std
    shapes = rng.normal(1.0, 0.05, (200, 62))
    std_cl = exclude_masked_std(shapes, cmask)
    ml = motion_level_with_mask(std_cl, True)
    print("Masked std mean: " + str(round(ml, 6)), flush=True)

    print("component_4 OK", flush=True)
