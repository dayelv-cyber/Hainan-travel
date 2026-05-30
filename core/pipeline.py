from __future__ import annotations

from pathlib import Path

import pandas as pd

from config import DEFAULT_CLASSES, DEFAULT_PURE_FILE
from core.aggregate import aggregate_replicates, comprehensive_advice
from core.io_loader import normalize_frequency, read_table, spectra_matrix, validation_warnings
from core.model_interface import load_existing_result, predict_mock_rules, predict_with_joblib
from core.peak_rules import evaluate_peak_rules
from core.preprocess import align_to_grid, preprocess_for_model, target_grid_from_pure
from core.similarity import build_pure_references, most_similar


def prepare_input(source, pure_source: str | Path = DEFAULT_PURE_FILE) -> dict:
    raw_df = read_table(source)
    df = normalize_frequency(raw_df)
    pure_df = read_table(pure_source)
    grid = target_grid_from_pure(pure_df)
    X, names, freq = spectra_matrix(df)
    aligned_raw = align_to_grid(freq, X, grid)
    processed = preprocess_for_model(aligned_raw, grid)
    references, pure_meta = build_pure_references(pure_df, grid)
    return {
        "raw_df": df,
        "grid": grid,
        "names": names,
        "aligned_raw": aligned_raw,
        "processed": processed,
        "references": references,
        "pure_meta": pure_meta,
        "warnings": validation_warnings(df),
    }


def run_analysis(source, mode: str = "mock_rules", pure_source: str | Path = DEFAULT_PURE_FILE, existing_result=None, model_path=None, classes: list[str] | None = None) -> dict:
    classes = classes or DEFAULT_CLASSES
    data = prepare_input(source, pure_source)
    if mode == "existing_result":
        details = load_existing_result(existing_result)
    elif mode == "joblib":
        details = predict_with_joblib(model_path, data["names"], data["processed"])
    else:
        details, sim_table = predict_mock_rules(
            data["names"], data["grid"], data["aligned_raw"], data["processed"], data["references"], classes
        )
        data["similarity_table"] = sim_table

    summary = aggregate_replicates(details, classes)
    peak_labels = []
    sim_labels = []
    for sample in summary["样品号"]:
        idxs = [i for i, name in enumerate(data["names"]) if name.startswith(sample + "-") or name == sample]
        if not idxs:
            peak_labels.append("Unknown")
            sim_labels.append("Unknown")
            continue
        raw_mean = data["aligned_raw"][idxs].mean(axis=0)
        proc_mean = data["processed"][idxs].mean(axis=0)
        peak_labels.append(evaluate_peak_rules(data["grid"], raw_mean, classes)["label"])
        sim_labels.append(most_similar(proc_mean, data["references"])["label"])
    summary["峰位规则"] = peak_labels
    summary["最相似纯品"] = sim_labels
    summary["综合建议"] = [
        comprehensive_advice(row["SVM主判"], row["峰位规则"], row["最相似纯品"])
        for _, row in summary.iterrows()
    ]
    data["details"] = details
    data["summary"] = summary
    return data

