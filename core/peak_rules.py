from __future__ import annotations

import numpy as np

from config import CLASS_LABELS_CN, PEAK_RULES, PEAK_TOLERANCE


def _hit_peak(grid: np.ndarray, y: np.ndarray, peak: float, tol: float) -> tuple[bool, float, float]:
    mask = (grid >= peak - tol) & (grid <= peak + tol)
    if not mask.any():
        return False, float("nan"), 0.0
    idxs = np.where(mask)[0]
    idx = idxs[np.argmax(y[idxs])]
    strength = float(y[idx])
    return strength >= 0.35, float(grid[idx]), strength


def evaluate_peak_rules(grid: np.ndarray, y: np.ndarray, classes: list[str] | None = None) -> dict:
    if classes is None:
        classes = list(PEAK_RULES)
    y_norm = y.astype(float)
    y_norm = y_norm - np.nanmin(y_norm)
    max_val = np.nanmax(y_norm)
    if max_val > 0:
        y_norm = y_norm / max_val

    details = {}
    scores = {}
    for cls in classes:
        rule = PEAK_RULES.get(cls)
        if not rule:
            continue
        hits = []
        for peak in rule["peaks"]:
            hit, found, strength = _hit_peak(grid, y_norm, peak, PEAK_TOLERANCE)
            hits.append({"target": peak, "hit": hit, "found": found, "strength": strength})
        details[cls] = hits
        scores[cls] = sum(1 for h in hits if h["hit"]) / max(len(hits), 1)

    if not scores:
        return {"label": "Unknown", "score": 0.0, "details": details}
    best = max(scores, key=scores.get)
    if scores[best] == 0:
        best = "Unknown"
    return {"label": best, "score": float(scores.get(best, 0.0)), "details": details}


def peak_label_text(result: dict, lang: str = "zh") -> str:
    label = result.get("label", "Unknown")
    if lang == "zh":
        return CLASS_LABELS_CN.get(label, label)
    return label

