from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from config import DEFAULT_CLASSES, DEFAULT_EXISTING_RESULT, MINOR_PROB_THRESHOLD
from core.aggregate import mixture_text
from core.io_loader import parse_sample_column
from core.peak_rules import evaluate_peak_rules
from core.similarity import most_similar


def load_existing_result(path: str | Path = DEFAULT_EXISTING_RESULT) -> pd.DataFrame:
    return pd.read_csv(path)


def _sample_sort_key(name: str) -> tuple[int, str]:
    sample, _ = parse_sample_column(name)
    digits = "".join(ch for ch in sample if ch.isdigit())
    return int(digits or 0), sample


def _stable_rng(sample: str) -> np.random.Generator:
    digest = hashlib.sha256(sample.encode("utf-8")).hexdigest()
    seed = int(digest[:8], 16)
    return np.random.default_rng(seed)


def _peak_scores(peak_result: dict, classes: list[str]) -> dict[str, float]:
    scores = {}
    details = peak_result.get("details", {})
    for cls in classes:
        hits = details.get(cls, [])
        scores[cls] = sum(1 for item in hits if item.get("hit")) / max(len(hits), 1) if hits else 0.0
    return scores


def _similarity_scores(sim_result: dict, classes: list[str]) -> dict[str, float]:
    table = sim_result.get("table")
    if table is None or table.empty:
        return {cls: 0.0 for cls in classes}
    raw = {str(row["class"]): float(row["score"]) for _, row in table.iterrows()}
    vals = np.array([raw.get(cls, 0.0) for cls in classes], dtype=float)
    vals = vals - vals.min()
    if vals.max() > 0:
        vals = vals / vals.max()
    return {cls: float(vals[i]) for i, cls in enumerate(classes)}


def _mock_probabilities(sample: str, peak_result: dict, sim_result: dict, classes: list[str]) -> dict[str, float]:
    peak = _peak_scores(peak_result, classes)
    sim = _similarity_scores(sim_result, classes)
    rng = _stable_rng(sample)

    values = {}
    for cls in classes:
        jitter = float(rng.uniform(0.0, 0.035))
        values[cls] = 0.06 + 0.52 * sim.get(cls, 0.0) + 0.38 * peak.get(cls, 0.0) + jitter

    total = sum(values.values())
    return {cls: values[cls] / total for cls in classes}


def predict_mock_rules(names: list[str], grid: np.ndarray, aligned_raw: np.ndarray, processed: np.ndarray, references: dict[str, np.ndarray], classes: list[str] | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    classes = classes or DEFAULT_CLASSES
    rows = []
    sim_rows = []

    groups: dict[str, list[int]] = {}
    for i, name in enumerate(names):
        sample, _ = parse_sample_column(name)
        groups.setdefault(sample, []).append(i)

    sample_cache = {}
    for sample, idxs in sorted(groups.items(), key=lambda kv: _sample_sort_key(kv[0])):
        raw_mean = aligned_raw[idxs].mean(axis=0)
        proc_mean = processed[idxs].mean(axis=0)
        peak_result = evaluate_peak_rules(grid, raw_mean, classes)
        sim_result = most_similar(proc_mean, references)
        probs = _mock_probabilities(sample, peak_result, sim_result, classes)
        sample_cache[sample] = (peak_result, sim_result, probs)

    for i, name in enumerate(names):
        sample, _ = parse_sample_column(name)
        peak_result, sim_result, probs = sample_cache[sample]
        pred = max(probs, key=probs.get)
        row = {"Sample": name, "Predicted_Type": pred}
        for cls in classes:
            row[f"{cls}_Prob"] = probs[cls]
        row["Mixture_Type"] = mixture_text(pd.Series(row), classes, MINOR_PROB_THRESHOLD)
        row["Peak_Rule"] = peak_result["label"]
        row["Similarity_Label"] = sim_result["label"]
        row["Similarity_Score"] = sim_result["score"]
        rows.append(row)

        table = sim_result["table"].copy()
        if not table.empty:
            table.insert(0, "Sample", name)
            sim_rows.append(table)
    return pd.DataFrame(rows), pd.concat(sim_rows, ignore_index=True) if sim_rows else pd.DataFrame()


def predict_with_joblib(model_path: str | Path, names: list[str], processed: np.ndarray) -> pd.DataFrame:
    try:
        import joblib
    except Exception as exc:
        raise RuntimeError("joblib is not installed. Install requirements.txt first.") from exc
    model = joblib.load(model_path)
    scaler = model["scaler"]
    pca = model["pca"]
    svm = model["svm"]
    classes = list(model.get("classes", svm.classes_))
    X = pca.transform(scaler.transform(processed))
    probs = svm.predict_proba(X)
    pred = svm.predict(X)
    rows = []
    for i, name in enumerate(names):
        row = {"Sample": name, "Predicted_Type": pred[i]}
        for j, cls in enumerate(classes):
            row[f"{cls}_Prob"] = float(probs[i, j])
        row["Mixture_Type"] = mixture_text(pd.Series(row), classes)
        rows.append(row)
    return pd.DataFrame(rows)
