from __future__ import annotations

import math

import numpy as np
import pandas as pd

from config import FREQ_MAX, FREQ_MIN, SG_DERIV, SG_POLY, SG_WINDOW
from core.io_loader import normalize_frequency


def target_grid_from_pure(pure_df: pd.DataFrame) -> np.ndarray:
    pure = normalize_frequency(pure_df)
    grid = pure["THz"].to_numpy(dtype=float)
    mask = (grid >= FREQ_MIN) & (grid <= FREQ_MAX)
    return np.sort(grid[mask])


def align_to_grid(freq: np.ndarray, X: np.ndarray, target_grid: np.ndarray) -> np.ndarray:
    order = np.argsort(freq)
    freq_sorted = freq[order]
    X_sorted = X[:, order]
    aligned = []
    for row in X_sorted:
        aligned.append(np.interp(target_grid, freq_sorted, row))
    return np.asarray(aligned, dtype=float)


def snv(X: np.ndarray) -> np.ndarray:
    mean = X.mean(axis=1, keepdims=True)
    std = X.std(axis=1, keepdims=True)
    std[std == 0] = 1
    return (X - mean) / std


def _savgol_fallback(X: np.ndarray, x: np.ndarray, window: int, poly: int, deriv: int) -> np.ndarray:
    half = window // 2
    out = np.empty_like(X, dtype=float)
    for i, row in enumerate(X):
        padded_y = np.pad(row, (half, half), mode="edge")
        step = np.median(np.diff(x))
        padded_x = np.concatenate([
            x[0] - step * np.arange(half, 0, -1),
            x,
            x[-1] + step * np.arange(1, half + 1),
        ])
        vals = []
        for j in range(len(x)):
            xs = padded_x[j:j + window]
            ys = padded_y[j:j + window]
            coeff = np.polyfit(xs - x[j], ys, poly)
            if deriv == 0:
                vals.append(np.polyval(coeff, 0))
            else:
                d = np.polyder(coeff, deriv)
                vals.append(np.polyval(d, 0) / math.factorial(0))
        out[i] = vals
    return out


def sg_derivative(X: np.ndarray, grid: np.ndarray, window: int = SG_WINDOW, poly: int = SG_POLY, deriv: int = SG_DERIV) -> np.ndarray:
    try:
        from scipy.signal import savgol_filter

        delta = float(np.median(np.diff(grid)))
        return savgol_filter(X, window_length=window, polyorder=poly, deriv=deriv, delta=delta, axis=1)
    except Exception:
        return _savgol_fallback(X, grid, window, poly, deriv)


def preprocess_for_model(X: np.ndarray, grid: np.ndarray) -> np.ndarray:
    return sg_derivative(snv(X), grid)


def minmax_rows(X: np.ndarray) -> np.ndarray:
    lo = X.min(axis=1, keepdims=True)
    hi = X.max(axis=1, keepdims=True)
    denom = hi - lo
    denom[denom == 0] = 1
    return (X - lo) / denom

