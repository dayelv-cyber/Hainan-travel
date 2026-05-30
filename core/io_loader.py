from __future__ import annotations

import re
from pathlib import Path
from typing import BinaryIO

import numpy as np
import pandas as pd

from config import CM_TO_THZ


SAMPLE_RE = re.compile(r"^(?P<sample>S?\d+)-(?P<rep>\d+)$", re.IGNORECASE)


def read_table(source: str | Path | BinaryIO) -> pd.DataFrame:
    """Read a merged spectral table and auto-detect delimiter."""
    if hasattr(source, "read"):
        return pd.read_csv(source, sep=None, engine="python")
    path = Path(source)
    return pd.read_csv(path, sep=None, engine="python")


def normalize_frequency(df: pd.DataFrame) -> pd.DataFrame:
    """Rename first column to THz and convert cm^-1 input when needed."""
    out = df.copy()
    first = out.columns[0]
    freq = pd.to_numeric(out[first], errors="coerce")
    out = out.loc[freq.notna()].copy()
    freq = freq.loc[freq.notna()].astype(float)
    if freq.max() > 50:
        out.insert(0, "THz", freq.to_numpy() * CM_TO_THZ)
    else:
        out.insert(0, "THz", freq.to_numpy())
    if first != "THz":
        out = out.drop(columns=[first])
    for col in out.columns[1:]:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out = out.dropna(axis=0, how="any").sort_values("THz").reset_index(drop=True)
    return out


def get_spectrum_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c != "THz"]


def parse_sample_column(col: str) -> tuple[str, int]:
    m = SAMPLE_RE.match(str(col))
    if not m:
        return str(col), 1
    sample = m.group("sample").upper()
    if not sample.startswith("S"):
        sample = "S" + sample
    return sample, int(m.group("rep"))


def group_columns(df: pd.DataFrame) -> dict[str, list[str]]:
    groups: dict[str, list[str]] = {}
    for col in get_spectrum_columns(df):
        sample, rep = parse_sample_column(col)
        groups.setdefault(sample, []).append(col)
    for sample, cols in groups.items():
        groups[sample] = sorted(cols, key=lambda c: parse_sample_column(c)[1])
    return dict(sorted(groups.items(), key=lambda kv: int(re.search(r"\d+", kv[0]).group())))


def spectra_matrix(df: pd.DataFrame) -> tuple[np.ndarray, list[str], np.ndarray]:
    names = get_spectrum_columns(df)
    return df[names].to_numpy(dtype=float).T, names, df["THz"].to_numpy(dtype=float)


def validation_warnings(df: pd.DataFrame, freq_min: float = 1.0, freq_max: float = 4.0) -> list[str]:
    warnings = []
    if "THz" not in df.columns:
        warnings.append("缺少频率列 / Missing frequency axis.")
        return warnings
    if df["THz"].min() > freq_min or df["THz"].max() < freq_max:
        warnings.append(f"频率范围未完全覆盖 {freq_min}-{freq_max} THz / Frequency range does not fully cover window.")
    bad = [c for c in get_spectrum_columns(df) if not SAMPLE_RE.match(str(c))]
    if bad:
        warnings.append("存在非 S编号-重复号 命名列: " + ", ".join(bad[:6]))
    groups = group_columns(df)
    reps = {s: len(cols) for s, cols in groups.items()}
    if len(set(reps.values())) > 1:
        warnings.append("样品重复数不一致 / Replicate counts differ: " + str(reps))
    return warnings

