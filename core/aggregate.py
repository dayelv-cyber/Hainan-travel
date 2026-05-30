from __future__ import annotations

import pandas as pd

from config import MINOR_PROB_THRESHOLD
from core.io_loader import parse_sample_column


def mixture_text(prob_row: pd.Series, classes: list[str], threshold: float = MINOR_PROB_THRESHOLD) -> str:
    probs = sorted([(cls, float(prob_row.get(f"{cls}_Prob", 0))) for cls in classes], key=lambda x: x[1], reverse=True)
    if not probs:
        return "Unknown"
    top1, p1 = probs[0]
    if len(probs) > 1 and probs[1][1] > threshold:
        return f"{top1} dominant + {probs[1][0]} minor"
    return f"{top1} dominant"


def aggregate_replicates(detail_df: pd.DataFrame, classes: list[str]) -> pd.DataFrame:
    data = detail_df.copy()
    if "Sample" not in data.columns:
        raise ValueError("detail_df must contain Sample column")
    data["样品号"] = [parse_sample_column(x)[0] for x in data["Sample"]]
    prob_cols = [f"{c}_Prob" for c in classes if f"{c}_Prob" in data.columns]
    rows = []
    for sample, g in data.groupby("样品号", sort=False):
        avg = g[prob_cols].mean() if prob_cols else pd.Series(dtype=float)
        best = avg.idxmax().replace("_Prob", "") if not avg.empty else "Unknown"
        pred_counts = g["Predicted_Type"].value_counts() if "Predicted_Type" in g else pd.Series(dtype=int)
        consistency = f"{int(pred_counts.iloc[0])}/{len(g)}" if len(pred_counts) else f"0/{len(g)}"
        row = {
            "样品号": sample,
            "SVM主判": best,
            "模型置信度": float(avg.max()) if not avg.empty else 0.0,
            "重复数": len(g),
            "重复一致性": consistency,
            "混合谱型": mixture_text(avg, classes),
        }
        for c in classes:
            row[f"{c}_Prob"] = float(avg.get(f"{c}_Prob", 0.0))
        rows.append(row)
    return pd.DataFrame(rows)


def comprehensive_advice(svm: str, peak: str, sim: str) -> str:
    votes = [x for x in [svm, peak, sim] if x and x != "Unknown"]
    if not votes:
        return "不确定，建议复核"
    top = max(set(votes), key=votes.count)
    count = votes.count(top)
    if count >= 3:
        return f"{top} 高可信（三方一致）"
    if count == 2:
        return f"{top} 中等可信（两方一致）"
    return "不确定/疑似混合，建议复核"

