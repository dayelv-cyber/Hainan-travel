from __future__ import annotations

import re

import numpy as np
import pandas as pd

from core.io_loader import normalize_frequency, spectra_matrix
from core.preprocess import align_to_grid, preprocess_for_model


def label_from_pure_name(name: str) -> str:
    upper = str(name).upper()
    if "AMP" in upper:
        return "MAP"
    if "CAOX" in upper:
        return "COM"
    if "DCY" in upper:
        return "CYS"
    if "URIC" in upper:
        return "UA"
    return "Unknown"


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    if np.std(a) == 0 or np.std(b) == 0:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def build_pure_references(pure_df: pd.DataFrame, target_grid: np.ndarray) -> tuple[dict[str, np.ndarray], pd.DataFrame]:
    pure = normalize_frequency(pure_df)
    X, names, freq = spectra_matrix(pure)
    aligned = align_to_grid(freq, X, target_grid)
    processed = preprocess_for_model(aligned, target_grid)

    rows = []
    refs: dict[str, list[np.ndarray]] = {}
    for i, name in enumerate(names):
        cls = label_from_pure_name(name)
        refs.setdefault(cls, []).append(processed[i])
        rows.append({"spectrum": name, "class": cls})
    ref_mean = {cls: np.mean(values, axis=0) for cls, values in refs.items() if cls != "Unknown"}
    return ref_mean, pd.DataFrame(rows)


def most_similar(sample_y: np.ndarray, references: dict[str, np.ndarray]) -> dict:
    rows = []
    for cls, ref in references.items():
        cos = cosine(sample_y, ref)
        corr = pearson(sample_y, ref)
        rows.append({"class": cls, "cosine": cos, "pearson": corr, "score": (cos + corr) / 2})
    if not rows:
        return {"label": "Unknown", "score": 0.0, "table": pd.DataFrame()}
    table = pd.DataFrame(rows).sort_values("score", ascending=False).reset_index(drop=True)
    return {"label": str(table.loc[0, "class"]), "score": float(table.loc[0, "score"]), "table": table}

