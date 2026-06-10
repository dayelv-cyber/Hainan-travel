from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import nnls
from scipy.signal import savgol_filter
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from config import (
    DCP_BACKGROUND_OFFSET,
    DEFAULT_CLASSES,
    EXCLUDED_SAMPLES,
    DEFAULT_MEAN_FILE,
    DEFAULT_NNLS_RESULT,
    DEFAULT_PATIENT_FILE,
    DEFAULT_PURE_FILE,
    DEFAULT_VALIDATION_FILE,
    FREQ_MAX,
    FREQ_MIN,
    MINOR_PROB_THRESHOLD,
    SG_DERIV,
    SG_POLY,
    SG_WINDOW,
)


def _rewind_if_needed(source) -> None:
    if hasattr(source, "seek"):
        source.seek(0)


def _read_csv(path: str | Path) -> pd.DataFrame:
    if isinstance(path, pd.DataFrame):
        return path.copy()
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


def normalize_weights(w: np.ndarray) -> np.ndarray:
    w = np.maximum(w, 0)
    s = w.sum()
    if s > 0:
        return w / s
    return w


def residual_flag(relative_residual: float) -> str:
    if relative_residual < 0.05:
        return "拟合很好（可信）"
    if relative_residual < 0.15:
        return "一般（可能轻微偏差）"
    if relative_residual < 0.25:
        return "建议复核"
    return "很可能基底不匹配/异常样本"


def _preprocess_spectrum(X: np.ndarray) -> np.ndarray:
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


def build_mean_spectra(raw_source: str | Path) -> pd.DataFrame:
    raw = _read_csv(raw_source)
    freq = raw.iloc[:, 0]
    spectra = raw.iloc[:, 1:]
    groups: dict[str, list[str]] = {}
    for col in spectra.columns:
        sample = str(col).split("-")[0]
        groups.setdefault(sample, []).append(col)

    mean_df = pd.DataFrame({"wave_num": freq})
    for sample, cols in groups.items():
        mean_df[sample] = spectra[cols].mean(axis=1)
    return mean_df


def compute_nnls_result(
    pure_source: str | Path = DEFAULT_PURE_FILE,
    mean_source: str | Path | pd.DataFrame = DEFAULT_MEAN_FILE,
) -> tuple[pd.DataFrame, list[str]]:
    pure_df = _read_csv(pure_source)
    mix_df = _read_csv(mean_source)

    pure_w = pure_df.iloc[:, 0].to_numpy(dtype=float)
    mix_w = mix_df.iloc[:, 0].to_numpy(dtype=float)
    w_min = max(pure_w.min(), mix_w.min())
    w_max = min(pure_w.max(), mix_w.max())
    w_common = np.linspace(w_min, w_max, 300)

    def interp(w, X):
        order = np.argsort(w)
        w_sorted = w[order]
        X_sorted = X[order, :]
        return np.vstack([np.interp(w_common, w_sorted, X_sorted[:, i]) for i in range(X_sorted.shape[1])]).T

    pure_X = interp(pure_w, pure_df.iloc[:, 1:].to_numpy(dtype=float))
    mix_X = interp(mix_w, mix_df.iloc[:, 1:].to_numpy(dtype=float))

    materials: dict[str, list[np.ndarray]] = {}
    for i, col in enumerate(pure_df.columns[1:]):
        materials.setdefault(_material_from_name(col), []).append(pure_X[:, i])

    material_names = list(materials.keys())
    basis = [np.mean(np.array(spectra).T, axis=1) for spectra in materials.values()]
    B = np.array(basis).T

    rows = []
    for i, sample in enumerate(mix_df.columns[1:]):
        w, _ = nnls(B, mix_X[:, i])
        residual = np.linalg.norm(B @ w - mix_X[:, i])
        relative_residual = residual / max(np.linalg.norm(mix_X[:, i]), 1e-9)
        prob = normalize_weights(w)
        if "DCP" in material_names:
            dcp_idx = material_names.index("DCP")
            prob[dcp_idx] = max(0, prob[dcp_idx] - DCP_BACKGROUND_OFFSET)
            prob = normalize_weights(prob)
        row = {
            "Sample": sample,
            "拟合残差": float(relative_residual),
            "相对残差": float(relative_residual),
        }
        row.update({cls: float(value) for cls, value in zip(material_names, prob)})
        rows.append(row)

    result = pd.DataFrame(rows)
    result = result[~result["Sample"].astype(str).isin(EXCLUDED_SAMPLES)].reset_index(drop=True)
    classes = [c for c in result.columns if c not in {"Sample", "拟合残差", "相对残差"}]
    result["NNLS主成分"] = result[classes].idxmax(axis=1)
    result["NNLS置信度"] = result[classes].max(axis=1)
    result["NNLS成分构成"] = result.apply(lambda row: _format_components(row, classes), axis=1)
    result["NNLS前2位"] = result[classes].apply(lambda row: list(row.sort_values(ascending=False).index[:2]), axis=1)
    result["残差标记"] = result["相对残差"].map(residual_flag)
    return result, classes


def load_nnls_result(path: str | Path = DEFAULT_NNLS_RESULT) -> tuple[pd.DataFrame, list[str]]:
    df = _read_csv(path).copy()
    if "Sample" in df.columns:
        df = df[~df["Sample"].astype(str).isin(EXCLUDED_SAMPLES)].reset_index(drop=True)
    if "相对残差" not in df.columns and "拟合残差" in df.columns:
        df["相对残差"] = df["拟合残差"]
    if "拟合残差" not in df.columns and "相对残差" in df.columns:
        df["拟合残差"] = df["相对残差"]
    classes = [c for c in df.columns if c not in {"Sample", "拟合残差", "相对残差", "residual", "relative_residual"}]
    df["NNLS主成分"] = df[classes].idxmax(axis=1)
    df["NNLS置信度"] = df[classes].max(axis=1)
    df["NNLS成分构成"] = df.apply(lambda row: _format_components(row, classes), axis=1)
    df["NNLS前2位"] = df[classes].apply(lambda row: list(row.sort_values(ascending=False).index[:2]), axis=1)
    if "相对残差" in df.columns:
        df["残差标记"] = df["相对残差"].map(residual_flag)
    return df, classes


def build_pca_projection(
    pure_path: str | Path = DEFAULT_PURE_FILE,
    sample_source: str | Path | pd.DataFrame = DEFAULT_MEAN_FILE,
) -> tuple[pd.DataFrame, pd.DataFrame, float]:
    pure_X, pure_names, pure_freq = _load_spectra(pure_path)
    sample_X, sample_names, sample_freq = _load_spectra(sample_source)

    pure_X, pure_cut_freq = _cut_freq(pure_X, pure_freq)
    target = np.sort(pure_cut_freq)
    pure_X = _align_to_target(pure_cut_freq, pure_X, target)
    sample_X = _align_to_target(sample_freq, sample_X, target)

    X_pure = _preprocess_spectrum(pure_X)
    X_sample = _preprocess_spectrum(sample_X)
    y = np.array([_material_from_name(name) for name in pure_names])

    scaler = StandardScaler()
    X_pure_scaled = scaler.fit_transform(X_pure)
    X_sample_scaled = scaler.transform(X_sample)

    pca = PCA(n_components=2)
    pure_pca = pca.fit_transform(X_pure_scaled)
    sample_pca = pca.transform(X_sample_scaled)
    # Match the orientation of the original PCA.py output; PCA signs are arbitrary.
    pure_pca[:, 1] *= -1
    sample_pca[:, 1] *= -1

    pure_df = pd.DataFrame(
        {
            "PC1": pure_pca[:, 0],
            "PC2": pure_pca[:, 1],
            "类别": y,
            "名称": pure_names,
        }
    )
    sample_df = pd.DataFrame(
        {
            "PC1": sample_pca[:, 0],
            "PC2": sample_pca[:, 1],
            "类别": "Patient",
            "名称": sample_names,
        }
    )
    explained = float(np.sum(pca.explained_variance_ratio_))
    return pure_df, sample_df, explained


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


def _composition_status(row: pd.Series, classes: list[str]) -> str:
    values = row[classes].astype(float).sort_values(ascending=False)
    main = str(values.index[0])
    main_value = float(values.iloc[0])
    second_value = float(values.iloc[1]) if len(values) > 1 else 0.0
    gap = main_value - second_value
    if main_value < 0.40 or gap < 0.15:
        return f"{main} 混合成分明显"
    if main_value >= 0.60:
        return f"{main} 主成分明确"
    return f"{main} 为主，伴少量混合"


def build_component_accuracy(merged: pd.DataFrame, classes: list[str]) -> pd.DataFrame:
    rows = []
    for cls in classes:
        actual = merged[merged["详细成分代码集"].map(lambda codes: cls in codes)]
        detected = int(actual["NNLS前2位"].map(lambda top2: cls in set(top2)).sum())
        actual_n = len(actual)

        predicted = merged[merged["NNLS前2位"].map(lambda top2: cls in set(top2))]
        correct_pred = int(predicted["详细成分代码集"].map(lambda codes: cls in codes).sum())
        pred_n = len(predicted)

        rows.append(
            {
                "成分": cls,
                "送检出现数": actual_n,
                "前二覆盖数": detected,
                "检出率": f"{detected}/{actual_n}" if actual_n else "0/0",
                "预测出现数": pred_n,
                "预测正确数": correct_pred,
                "预测准确率": f"{correct_pred}/{pred_n}" if pred_n else "0/0",
            }
        )
    return pd.DataFrame(rows)


def build_validation(summary: pd.DataFrame, validation_path: str | Path = DEFAULT_VALIDATION_FILE, classes: list[str] | None = None) -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    classes = classes or DEFAULT_CLASSES
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
    component_accuracy = build_component_accuracy(merged, classes)

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
    return table, metrics, component_accuracy


def run_v2_analysis(
    nnls_path: str | Path = DEFAULT_NNLS_RESULT,
    pure_path: str | Path = DEFAULT_PURE_FILE,
    mean_path: str | Path = DEFAULT_MEAN_FILE,
    raw_path: str | Path = DEFAULT_PATIENT_FILE,
    validation_path: str | Path = DEFAULT_VALIDATION_FILE,
    uploaded_raw_source=None,
) -> dict:
    raw_source = uploaded_raw_source if uploaded_raw_source is not None else raw_path
    if raw_source is not None:
        mean_df = build_mean_spectra(raw_source)
    else:
        mean_df = _read_csv(mean_path)
    nnls, nnls_classes = compute_nnls_result(pure_path, mean_df)
    summary = nnls.copy()
    summary["构成状态"] = summary.apply(lambda row: _composition_status(row, nnls_classes), axis=1)
    summary["综合判断"] = np.where(
        summary["相对残差"] >= 0.15,
        summary["构成状态"] + "，" + summary["残差标记"],
        summary["构成状态"] + "，以NNLS分解为主",
    )
    summary["S69提示"] = summary["Sample"].map(
        lambda x: "疑似磷酸铵镁(AMP), 基底未覆盖, 结果不可靠, 仅供参考" if str(x) == "S69" else ""
    )
    summary = summary.rename(columns={"Sample": "样品"})

    validation_table, validation_metrics, component_accuracy = build_validation(summary, validation_path, nnls_classes)
    pca_pure, pca_sample, pca_explained = build_pca_projection(pure_path, mean_df)
    raw_df = _read_csv(raw_path)

    return {
        "summary": summary,
        "validation": validation_table,
        "validation_metrics": validation_metrics,
        "component_accuracy": component_accuracy,
        "classes": nnls_classes,
        "mean_df": mean_df,
        "raw_df": raw_df,
        "pca_pure": pca_pure,
        "pca_sample": pca_sample,
        "pca_explained_variance": pca_explained,
    }
