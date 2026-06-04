from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from config import (
    DEFAULT_CLASSES,
    DEFAULT_MEAN_FILE,
    DEFAULT_NNLS_RESULT,
    DEFAULT_PATIENT_FILE,
    DEFAULT_PURE_FILE,
    DEFAULT_VALIDATION_FILE,
    FREQ_MAX,
    FREQ_MIN,
    MINOR_PROB_THRESHOLD,
    PCA_COMPONENTS,
    SG_DERIV,
    SG_POLY,
    SG_WINDOW,
)


def _rewind_if_needed(source) -> None:
    if hasattr(source, "seek"):
        source.seek(0)


def _read_csv(path: str | Path) -> pd.DataFrame:
    _rewind_if_needed(path)
    return pd.read_csv(path)


def _load_spectra(path: str | Path) -> tuple[np.ndarray, list[str], np.ndarray]:
    df = _read_csv(path)
    freq = df.iloc[:, 0].to_numpy(dtype=float)
    names = list(df.columns[1:])
    X = df.iloc[:, 1:].to_numpy(dtype=float).T
    return X, names, freq


def _material_from_name(name: str) -> str:
    return str(name).split("-")[0].upper()


def _cut_freq(X: np.ndarray, freq: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mask = (freq >= FREQ_MIN) & (freq <= FREQ_MAX)
    return X[:, mask], freq[mask]


def _align_to_target(freq: np.ndarray, X: np.ndarray, target: np.ndarray) -> np.ndarray:
    order = np.argsort(freq)
    freq_sorted = freq[order]
    X_sorted = X[:, order]
    aligned = np.vstack([np.interp(target, freq_sorted, row) for row in X_sorted])
    return aligned


def _snv(X: np.ndarray) -> np.ndarray:
    mean = X.mean(axis=1, keepdims=True)
    std = X.std(axis=1, keepdims=True)
    std = np.where(std == 0, 1, std)
    return (X - mean) / std


def _preprocess_for_svm(X: np.ndarray) -> np.ndarray:
    X = _snv(X)
    return savgol_filter(X, window_length=SG_WINDOW, polyorder=SG_POLY, deriv=SG_DERIV, axis=1)


def _format_components(row: pd.Series, classes: list[str], threshold: float = MINOR_PROB_THRESHOLD) -> str:
    parts = []
    for cls in sorted(classes, key=lambda c: float(row[c]), reverse=True):
        value = float(row[cls])
        if value >= threshold:
            parts.append(f"{cls} {value:.2f}")
    if not parts:
        top = max(classes, key=lambda c: float(row[c]))
        parts.append(f"{top} {float(row[top]):.2f}")
    return " / ".join(parts)


def load_nnls_result(path: str | Path = DEFAULT_NNLS_RESULT) -> tuple[pd.DataFrame, list[str]]:
    df = _read_csv(path).copy()
    classes = [c for c in df.columns if c != "Sample"]
    df["NNLS主成分"] = df[classes].idxmax(axis=1)
    df["NNLS置信度"] = df[classes].max(axis=1)
    df["NNLS成分构成"] = df.apply(lambda row: _format_components(row, classes), axis=1)
    df["NNLS前2位"] = df[classes].apply(lambda row: list(row.sort_values(ascending=False).index[:2]), axis=1)
    return df, classes


def train_predict_svm(
    pure_path: str | Path = DEFAULT_PURE_FILE,
    sample_path: str | Path = DEFAULT_MEAN_FILE,
) -> pd.DataFrame:
    pure_X, pure_names, pure_freq = _load_spectra(pure_path)
    sample_X, sample_names, sample_freq = _load_spectra(sample_path)

    pure_X, pure_cut_freq = _cut_freq(pure_X, pure_freq)
    target = np.sort(pure_cut_freq)
    pure_X = _align_to_target(pure_cut_freq, pure_X, target)
    sample_X = _align_to_target(sample_freq, sample_X, target)

    X_pure = _preprocess_for_svm(pure_X)
    X_sample = _preprocess_for_svm(sample_X)
    y = np.array([_material_from_name(name) for name in pure_names])

    scaler = StandardScaler()
    X_pure_scaled = scaler.fit_transform(X_pure)
    X_sample_scaled = scaler.transform(X_sample)

    n_components = min(PCA_COMPONENTS, X_pure_scaled.shape[0] - 1, X_pure_scaled.shape[1])
    pca = PCA(n_components=n_components)
    X_pure_pca = pca.fit_transform(X_pure_scaled)
    X_sample_pca = pca.transform(X_sample_scaled)

    clf = SVC(kernel="rbf", probability=True, random_state=42)
    clf.fit(X_pure_pca, y)

    probs = clf.predict_proba(X_sample_pca)
    pred = clf.predict(X_sample_pca)
    classes = list(clf.classes_)

    out = pd.DataFrame({"Sample": sample_names, "SVM主判": pred})
    out["SVM置信度"] = probs.max(axis=1)
    for i, cls in enumerate(classes):
        out[f"SVM_{cls}"] = probs[:, i]
    out.attrs["pca_explained_variance"] = float(np.sum(pca.explained_variance_ratio_))
    return out


def _coarse_to_code(value) -> str:
    text = str(value).strip()
    mapping = {
        "草酸钙": "CAOX",
        "尿酸": "URIC",
        "磷灰石": "APA",
        "AMP": "AMP",
    }
    return mapping.get(text, "Unknown")


def _detail_to_codes(value) -> set[str]:
    text = str(value)
    codes: set[str] = set()
    if "草酸钙" in text:
        codes.add("CAOX")
    if "胱氨酸" in text:
        codes.add("DCY")
    if "尿酸" in text or "黄嘌呤" in text:
        codes.add("URIC")
    if "磷灰石" in text:
        codes.add("APA")
    if "磷酸氢钙" in text:
        codes.add("DCP")
    if "磷酸铵镁" in text or "AMP" in text:
        codes.add("AMP")
    return codes


def _yes_no(value: bool) -> str:
    return "✓" if value else "✗"


def build_validation(summary: pd.DataFrame, validation_path: str | Path = DEFAULT_VALIDATION_FILE) -> tuple[pd.DataFrame, dict]:
    _rewind_if_needed(validation_path)
    raw = pd.read_excel(validation_path, sheet_name="Sheet1")
    validation = raw[["序号", "上理工", "长海"]].copy()
    validation["粗标签代码"] = validation["上理工"].map(_coarse_to_code)
    validation["详细成分代码集"] = validation["长海"].map(_detail_to_codes)

    merged = summary.merge(validation, left_on="样品", right_on="序号", how="left")

    def strict_hit(row) -> bool:
        return row["NNLS主成分"] == row["粗标签代码"]

    def reasonable_hit(row) -> bool:
        return row["NNLS主成分"] in row["详细成分代码集"]

    def combo_hit(row) -> bool:
        return bool(set(row["NNLS前2位"]) & set(row["详细成分代码集"]))

    merged["严格命中"] = merged.apply(strict_hit, axis=1)
    merged["合理命中"] = merged.apply(reasonable_hit, axis=1)
    merged["组合命中"] = merged.apply(combo_hit, axis=1)
    merged["详细成分代码"] = merged["详细成分代码集"].map(lambda x: "/".join(sorted(x)) if isinstance(x, set) else "")
    merged["S69提示"] = merged["样品"].map(
        lambda x: "疑似磷酸铵镁(AMP), 基底未覆盖, 结果不可靠, 仅供参考" if str(x) == "S69" else ""
    )

    total = len(merged)
    metrics = {
        "strict": int(merged["严格命中"].sum()),
        "reasonable": int(merged["合理命中"].sum()),
        "combo": int(merged["组合命中"].sum()),
        "total": total,
    }

    table = merged[
        [
            "样品",
            "NNLS主成分",
            "NNLS成分构成",
            "上理工",
            "长海",
            "详细成分代码",
            "严格命中",
            "合理命中",
            "组合命中",
            "S69提示",
        ]
    ].copy()
    for col in ["严格命中", "合理命中", "组合命中"]:
        table[col] = table[col].map(_yes_no)
    return table, metrics


def run_v2_analysis(
    nnls_path: str | Path = DEFAULT_NNLS_RESULT,
    pure_path: str | Path = DEFAULT_PURE_FILE,
    mean_path: str | Path = DEFAULT_MEAN_FILE,
    raw_path: str | Path = DEFAULT_PATIENT_FILE,
    validation_path: str | Path = DEFAULT_VALIDATION_FILE,
) -> dict:
    nnls, nnls_classes = load_nnls_result(nnls_path)
    svm = train_predict_svm(pure_path, mean_path)
    summary = nnls.merge(svm, on="Sample", how="left")

    summary["双法是否一致"] = np.where(summary["NNLS主成分"] == summary["SVM主判"], "✓ 一致", "✗ 不一致")
    summary["综合判断"] = summary.apply(
        lambda row: (
            f"{row['NNLS主成分']} 为主(双法印证,高可信)"
            if row["NNLS主成分"] == row["SVM主判"]
            else "疑似混合,建议复核"
        ),
        axis=1,
    )
    summary["S69提示"] = summary["Sample"].map(
        lambda x: "疑似磷酸铵镁(AMP), 基底未覆盖, 结果不可靠, 仅供参考" if str(x) == "S69" else ""
    )
    summary = summary.rename(columns={"Sample": "样品"})

    validation_table, validation_metrics = build_validation(summary, validation_path)
    mean_df = _read_csv(mean_path)
    raw_df = _read_csv(raw_path)

    return {
        "summary": summary,
        "validation": validation_table,
        "validation_metrics": validation_metrics,
        "classes": nnls_classes,
        "mean_df": mean_df,
        "raw_df": raw_df,
        "svm_pca_explained_variance": svm.attrs.get("pca_explained_variance"),
    }
