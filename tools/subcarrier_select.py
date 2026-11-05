"""Subcarrier sensitivity selection for WiFi Space Mapper.

Ported from v2/crates/wifi-densepose-signal/src/subcarrier_selection.rs
(RuView, MIT/Apache-2.0). Identifies the most motion-sensitive subcarriers
by computing per-subcarrier variance ratios.

Two methods:
  1. select_sensitive_subcarriers() -- motion-variance / static-variance ratio
     (requires separate still + motion datasets; higher accuracy)
  2. select_by_variance() -- top-K by absolute variance (no separate dataset needed)

Usage::
    # Method 1: variance ratio
    sel = select_sensitive_subcarriers(motion_data, static_data, top_k=20)

    # Method 2: plain variance ranking
    sel = select_by_variance(all_data, top_k=20)
"""

import numpy as np


def column_variance(data, col):
    """Variance of a single column in a 2-D matrix."""
    col_data = data[:, col]
    n = len(col_data)
    if n < 2:
        return 0.0
    mean = col_data.mean()
    return float(((col_data - mean) ** 2).sum() / (n - 1))


class SubcarrierSelection:
    """Result of subcarrier selection.
    
    Attributes
        selected_indices : list[int]  Selected subcarrier indices (sorted by score desc.)
        sensitivity_scores : list[float]  Scores for ALL subcarriers
    """
    def __init__(self, selected_indices=None, sensitivity_scores=None):
        self.selected_indices = selected_indices or []
        self.sensitivity_scores = sensitivity_scores or []

    @property
    def sensitivity_map(self):
        """Dict mapping subcarrier index -> sensitivity score."""
        return dict(zip(range(len(self.sensitivity_scores)), self.sensitivity_scores))

    def plot_bars(self, ax=None, max_display=20):
        """Plot bar chart of sensitivity scores.
    
    Parameters
        ax : matplotlib Axes (optional)
        max_display : int  Show only top N
    
    Returns
        matplotlib Axes
        """
        if ax is None:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(max_display / 2, 3))
        scores = self.sensitivity_scores
        idx = list(range(len(scores)))
        bars = ax.bar(idx, scores, color='#6e4aff', alpha=0.7)
        ax.set_xlabel('Subcarrier index')
        ax.set_ylabel('Sensitivity score')
        ax.set_title(f'Subcarrier sensitivity ({len(self.selected_indices)} selected)')
        # Highlight selected ones
        for i, bar in enumerate(bars):
            if i in self.selected_indices:
                bar.set_alpha(1.0)
                bar.set_edgecolor('#39d0d8')
                bar.set_linewidth(2)
        return ax


def select_sensitive_subcarriers(motion_data, static_data, top_k=20, min_sensitivity=1.5):
    """Rank subcarriers by variance ratio (motion / static) and select top-K.

    Parameters
        motion_data : ndarray[T, C]  Amplitude frames during motion periods
        static_data : ndarray[T, C]  Amplitude frames during still periods
        top_k : int  Number of subcarriers to select
        min_sensitivity : float  Minimum sensitivity ratio to include

    Returns
        SubcarrierSelection with selected indices and full sensitivity scores
        """
    motion_data = np.asarray(motion_data, dtype=float)
    static_data = np.asarray(static_data, dtype=float)
    _, n_sc_motion = motion_data.shape
    _, n_sc_static = static_data.shape

    if n_sc_motion != n_sc_static:
        raise ValueError(f"Subcarrier count mismatch: motion={n_sc_motion}, static={n_sc_static}")
    if n_sc_motion == 0:
        raise ValueError("No subcarriers in input")

    scores = []
    for k in range(n_sc_motion):
        motion_var = column_variance(motion_data, k)
        static_var = column_variance(static_data, k)
        sensitivity = motion_var / (static_var + 1e-12)
        scores.append(sensitivity)

    # Rank by sensitivity descending
    ranked = sorted(range(n_sc_motion), key=lambda i: scores[i], reverse=True)
    selected = [idx for idx in ranked if scores[idx] >= min_sensitivity][:top_k]
    selected.sort()  # sort ascending for consistent ordering

    return SubcarrierSelection(selected_indices=selected, sensitivity_scores=scores)


def select_by_variance(all_data, top_k=20):
    """Rank subcarriers by absolute variance (no separate static period needed).

    Good for quick analysis when you cannot separate still/motion periods.
    Less precise than variance-ratio but requires only one dataset.

    Parameters
        all_data : ndarray[T, C]  All amplitude frames (mixed states)
        top_k : int  Number of subcarriers to select

    Returns
        SubcarrierSelection with selected indices and variance scores
        """
    all_data = np.asarray(all_data, dtype=float)
    _, n_sc = all_data.shape
    scores = [column_variance(all_data, k) for k in range(n_sc)]
    ranked = sorted(range(n_sc), key=lambda i: scores[i], reverse=True)
    selected = ranked[:top_k]
    selected.sort()

    return SubcarrierSelection(selected_indices=selected, sensitivity_scores=scores)


def mask_sensitivity_for_plotting(sel, n_display=64, highlight_width=1):
    """Create a display-friendly array highlighting sensitive subcarriers.

    Useful for plotting a line where sensitive bins stand out visually.

    Parameters
        sel : SubcarrierSelection
        n_display : int  Number of subcarriers shown (usually 64)
        highlight_width : int  How many bands around each selected index to show

    Returns
        ndarray[n_display]  Base amplitude with sensitive regions elevated
        """
    result = np.zeros(n_display)
    for sc in sel.selected_indices:
        for offset in range(-highlight_width, highlight_width + 1):
            idx = sc + offset
            if 0 <= idx < n_display:
                result[idx] = max(result[idx], sel.sensitivity_scores[sc])
    return result


if __name__ == "__main__":
    """Quick smoke test using synthetic data."""
    rng = np.random.default_rng(42)

    # Create fake still + motion data for testing
    n_still, n_motion, n_sc, n_frames = 500, 500, 62, 200
    static = rng.normal(1.0, 0.05, (n_still, n_sc))
    motion = rng.normal(1.0 + 0.3 * np.arange(n_sc) * 0.02, 0.15, (n_motion, n_sc))

    sel = select_sensitive_subcarriers(motion, static, top_k=20)
    print(f"Selected {len(sel.selected_indices)} subcarriers of {n_sc}", flush=True)
    print(f"Top-5 scores: {sel.sensitivity_scores[:5]}")
    assert len(sel.selected_indices) > 0, "Should have selected at least one"
    print("subcarrier_select OK", flush=True)