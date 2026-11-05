"""Hampel outlier filter for CSI amplitude vectors.

Port of v2/crates/wifi-densepose-signal/src/hampel.rs from RuView (MIT / MIT-OR-Apache-2.0).

Uses running median and MAD (Median Absolute Deviation) instead of mean/std, making it
resistant to up to 50% contamination unlike Z-score methods where outliers corrupt the
mean and mask themselves.

Why on CSI: A single noisy sample can inflate the per-subcarrier moving-window std by an
amount that alone pushes the motion level over the calibrated threshold. The Hampel filter
replaces those spikes with the local median before gain removal and variance computation,
so one bad sample cannot masquerade as sustained motion.
"""
import numpy as np


MAD_SCALE = 1.4826
ZERO_MAD_EPSILON = 1e-15


def _median_1d(data):
    """Median of a list of floats (sorts a copy)."""
    if len(data) == 0:
        return 0.0
    s = sorted(data)
    n = len(s)
    mid = n // 2
    if n % 2 == 0:
        return (s[mid - 1] + s[mid]) / 2.0
    return s[mid]

def _mad(data, med):
    """Median absolute deviation given precomputed median."""
    devs = [abs(x - med) for x in data]
    return _median_1d(devs)

def hampel_filter_1d(signal, half_window=3, threshold=3.0):
    """Apply a Hampel filter to a 1-D signal array.

    For each sample, computes the median and MAD of the surrounding window.
    If the sample deviates from the median by more than ``threshold * sigma_est``,
    it is replaced with the median value.

    Returns (filtered, outlier_mask, medians).
    """
    sig = np.asarray(signal, dtype=float)
    n = len(sig)
    if n == 0:
        return np.array([], dtype=float), np.array([], dtype=bool), np.array([], dtype=float)
    filtered = sig.copy()
    outlier_mask = np.zeros(n, dtype=bool)
    medians = np.empty(n)
    hw = min(half_window, n // 2)
    for i in range(n):
        start = max(0, i - hw)
        end = min(n, i + hw + 1)
        window = sig[start:end]
        med = _median_1d(window)
        mad_val = _mad(window, med)
        sigma_est = MAD_SCALE * mad_val
        medians[i] = med
        deviation = abs(sig[i] - med)
        if sigma_est > ZERO_MAD_EPSILON:
            is_outlier = deviation > threshold * sigma_est
        else:
            is_outlier = deviation > ZERO_MAD_EPSILON
        if is_outlier:
            filtered[i] = med
            outlier_mask[i] = True
    return filtered, outlier_mask, medians

def hampel_filter_column(matrix, col_idx, half_window=3, threshold=3.0):
    """Hampel filter a single column of a 2-D matrix (T frames x C subcarriers).
    Returns (filtered_col, outlier_mask).
    """
    return hampel_filter_1d(matrix[:, col_idx], half_window, threshold)

if __name__ == "__main__":
    signal = [1.0] * 50
    signal[25] = 100.0
    filt, mask, _ = hampel_filter_1d(signal)
    assert mask[25], "Should detect spike"
    assert abs(filt[25] - 1.0) < 1e-10, "Should replace spike with median"
    print("hampel OK")