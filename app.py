from __future__ import annotations

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))

import pandas as pd

try:
    import streamlit as st
except Exception as exc:  # pragma: no cover
    raise RuntimeError("Streamlit is not installed. Run: pip install -r requirements.txt") from exc

from config import (
    APP_NAME_CN,
    APP_NAME_EN,
    CLASS_LABELS_CN,
    CLASS_LABELS_EN,
    DEFAULT_CLASSES,
    DEFAULT_EXISTING_RESULT,
    DEFAULT_PATIENT_FILE,
    DEFAULT_PURE_FILE,
    THERAPY_NOTES_CN,
    THERAPY_NOTES_EN,
)
from core.export import export_excel
from core.pipeline import run_analysis
from i18n import t


def class_name(cls: str, lang: str) -> str:
    labels = CLASS_LABELS_CN if lang == "zh" else CLASS_LABELS_EN
    return f"{cls} {labels.get(cls, '')}".strip()


def mixture_display(text: str, lang: str) -> str:
    if lang != "zh" or not isinstance(text, str):
        return text
    out = text
    for cls, name in CLASS_LABELS_CN.items():
        out = out.replace(cls, f"{cls}({name})")
    out = out.replace(" dominant", "为主")
    out = out.replace(" minor", "次要")
    out = out.replace(" + ", " + ")
    return out


def localized_table(df: pd.DataFrame, lang: str) -> pd.DataFrame:
    out = df.copy()
    if lang == "zh":
        rename = {
            "Sample": "光谱编号",
            "Predicted_Type": "预测类别",
            "Mixture_Type": "混合谱型",
            "Peak_Rule": "峰位规则",
            "Similarity_Label": "最相似纯品",
            "Similarity_Score": "相似度得分",
        }
        out = out.rename(columns=rename)
        for col in ["预测类别", "峰位规则", "最相似纯品", "SVM主判"]:
            if col in out.columns:
                out[col] = out[col].map(lambda x: class_name(str(x), "zh") if pd.notna(x) else x)
        if "混合谱型" in out.columns:
            out["混合谱型"] = out["混合谱型"].map(lambda x: mixture_display(x, "zh"))
    return out


st.set_page_config(page_title=APP_NAME_CN, layout="wide")

st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(135deg, #f7f8fb 0%, #ffffff 58%, #fff5f5 100%);
    }
    .hero {
        padding: 18px 22px;
        border-left: 6px solid #B22222;
        background: #fff;
        border-radius: 8px;
        box-shadow: 0 8px 24px rgba(26,43,74,.08);
    }
    .small-note { color: #6b7280; font-size: 13px; }
    </style>
    """,
    unsafe_allow_html=True,
)

lang_label = st.sidebar.radio("Language / 语言", ["中文", "English"], horizontal=True)
lang = "zh" if lang_label == "中文" else "en"

title = APP_NAME_CN if lang == "zh" else APP_NAME_EN
st.markdown(f"<div class='hero'><h1>{title}</h1><p>{t(lang, 'subtitle')}</p></div>", unsafe_allow_html=True)
st.warning(t(lang, "disclaimer"))

st.sidebar.header("分析" if lang == "zh" else "Analysis")
mode_label = st.sidebar.selectbox(
    t(lang, "mode"),
    ["mock-规则 / Rule mock", "mock-读现成结果 / Existing result", "真实 joblib / Real model"],
)
uploaded = st.sidebar.file_uploader(t(lang, "upload"), type=["csv", "txt"])
use_demo = st.sidebar.checkbox(t(lang, "demo"), value=uploaded is None)
model_path = st.sidebar.text_input("joblib 模型路径" if lang == "zh" else "joblib model path", value="")

mode_key = "mock_rules"
if "Existing result" in mode_label:
    mode_key = "existing_result"
elif "Real model" in mode_label:
    mode_key = "joblib"

if mode_key == "existing_result":
    source = DEFAULT_PATIENT_FILE
    source_name = DEFAULT_PATIENT_FILE.name
    source_kind = "Senior demo CSV"
else:
    source = uploaded if uploaded is not None else DEFAULT_PATIENT_FILE
    source_name = uploaded.name if uploaded is not None else DEFAULT_PATIENT_FILE.name
    source_kind = "Uploaded CSV" if uploaded is not None else "Demo CSV"

input_signature = (mode_key, source_name, str(model_path))
if st.session_state.get("input_signature") != input_signature:
    st.session_state.pop("analysis", None)
    st.session_state.pop("analysis_meta", None)
    st.session_state.input_signature = input_signature

run = st.sidebar.button(t(lang, "run"), type="primary")
if "analysis" not in st.session_state and use_demo:
    run = True

if run:
    with st.spinner("Analyzing spectra..." if lang == "en" else "正在分析光谱..."):
        if mode_key == "existing_result":
            result = run_analysis(
                source,
                mode="existing_result",
                pure_source=DEFAULT_PURE_FILE,
                existing_result=DEFAULT_EXISTING_RESULT,
                classes=DEFAULT_CLASSES,
            )
        elif mode_key == "joblib":
            result = run_analysis(
                source,
                mode="joblib",
                pure_source=DEFAULT_PURE_FILE,
                model_path=model_path,
                classes=DEFAULT_CLASSES,
            )
        else:
            result = run_analysis(source, mode="mock_rules", pure_source=DEFAULT_PURE_FILE, classes=DEFAULT_CLASSES)

        st.session_state.analysis = result
        st.session_state.analysis_meta = {
            "mode_key": mode_key,
            "mode_label": mode_label,
            "source_name": source_name,
            "source_kind": source_kind,
        }

if "analysis" not in st.session_state:
    st.info("请上传数据或使用默认演示数据，然后点击“运行分析”。" if lang == "zh" else "Upload data or use demo data, then click Run analysis.")
    st.stop()

analysis = st.session_state.analysis
summary = analysis["summary"]
details = analysis["details"]
grid = analysis["grid"]
names = analysis["names"]
meta = st.session_state.get("analysis_meta", {})

cols = st.columns(4)
cols[0].metric("Samples" if lang == "en" else "样品数", len(summary))
cols[1].metric("Spectra" if lang == "en" else "光谱数", len(details))
cols[2].metric("Grid" if lang == "en" else "频率网格", len(grid))
cols[3].metric("Window", f"{grid.min():.2f}-{grid.max():.2f} THz")

if lang == "zh":
    st.info(f"当前数据源：{meta.get('source_kind', '')} / {meta.get('source_name', '')}；当前分析模式：{meta.get('mode_label', '')}")
    if meta.get("mode_key") == "mock_rules":
        st.warning("当前使用 mock-规则模式：结果来自峰位规则 + 纯品相似度模拟概率，不是学长 SVM 模型输出。")
    elif meta.get("mode_key") == "existing_result":
        st.warning("当前使用学长现成结果 CSV：该模式固定读取“病人结石混合谱型分析.csv”，不会重新预测上传文件。")
else:
    st.info(f"Data source: {meta.get('source_kind', '')} / {meta.get('source_name', '')}; mode: {meta.get('mode_label', '')}")
    if meta.get("mode_key") == "mock_rules":
        st.warning("Rule-mock mode: probabilities are simulated from peak rules and pure-reference similarity, not from the senior SVM model.")
    elif meta.get("mode_key") == "existing_result":
        st.warning("Existing-result mode reads 病人结石混合谱型分析.csv and does not predict the uploaded file.")

for warning in analysis.get("warnings", []):
    st.warning(warning)

st.markdown("### " + ("本批样品初筛概览" if lang == "zh" else "Batch Screening Overview"))
dist = summary["SVM主判"].value_counts().reindex(DEFAULT_CLASSES, fill_value=0).reset_index()
dist.columns = ["类别" if lang == "zh" else "Class", "样品数" if lang == "zh" else "Samples"]
if lang == "zh":
    dist["显示类别"] = dist["类别"].map(lambda x: class_name(str(x), "zh"))
    st.bar_chart(dist, x="显示类别", y="样品数", height=220)
    top_cls = summary["SVM主判"].value_counts().idxmax()
    top_n = int(summary["SVM主判"].value_counts().max())
    uncertain = int(summary["综合建议"].astype(str).str.contains("不确定").sum()) if "综合建议" in summary else 0
    st.caption(f"本批 {len(summary)} 例样品中，{class_name(top_cls, 'zh')} 主导 {top_n} 例；不确定/需复核 {uncertain} 例。计数基于样品级主判，不代表真实化学百分比。")
else:
    st.bar_chart(dist, x="Class", y="Samples", height=220)

clinical_tab = "临床参考" if lang == "zh" else "Clinical reference"
tab1, tab2, tab3, tab4 = st.tabs([t(lang, "summary"), t(lang, "details"), t(lang, "spectrum"), clinical_tab])

with tab1:
    st.dataframe(localized_table(summary, lang), use_container_width=True)
    if st.button(t(lang, "export")):
        path = export_excel(summary, details, analysis.get("similarity_table"))
        st.success(str(path))

with tab2:
    st.dataframe(localized_table(details, lang), use_container_width=True)

with tab3:
    sample_col = "样品号"
    sample = st.selectbox(t(lang, "sample"), list(summary[sample_col]))
    idxs = [i for i, name in enumerate(names) if str(name).startswith(sample + "-") or str(name) == sample]
    if idxs:
        raw_mean = analysis["aligned_raw"][idxs].mean(axis=0)
        proc_mean = analysis["processed"][idxs].mean(axis=0)
        raw_label = "原始/对齐光谱" if lang == "zh" else "Raw/aligned"
        proc_label = "预处理后二阶导数" if lang == "zh" else "Preprocessed derivative"
        raw_df = pd.DataFrame({"THz": grid, raw_label: raw_mean})
        proc_df = pd.DataFrame({"THz": grid, proc_label: proc_mean})
        st.caption("原始光谱和预处理后二阶导数量纲不同，分开显示。" if lang == "zh" else "Raw and preprocessed spectra use different scales and are shown separately.")
        st.line_chart(raw_df, x="THz", y=raw_label, height=250)
        st.line_chart(proc_df, x="THz", y=proc_label, height=250)
        row = summary.loc[summary[sample_col] == sample].iloc[0]
        prob_cols = [c for c in summary.columns if c.endswith("_Prob")]
        if lang == "zh":
            prob_df = pd.DataFrame({"类别": [class_name(c.replace("_Prob", ""), "zh") for c in prob_cols], "概率": [row[c] for c in prob_cols]})
            st.bar_chart(prob_df, x="类别", y="概率", height=260)
        else:
            prob_df = pd.DataFrame({"Class": [c.replace("_Prob", "") for c in prob_cols], "Probability": [row[c] for c in prob_cols]})
            st.bar_chart(prob_df, x="Class", y="Probability", height=260)

with tab4:
    labels = CLASS_LABELS_CN if lang == "zh" else CLASS_LABELS_EN
    notes = THERAPY_NOTES_CN if lang == "zh" else THERAPY_NOTES_EN
    therapy = pd.DataFrame(
        [{"Class": cls, "Name": labels.get(cls, cls), "Reference": notes.get(cls, "")} for cls in DEFAULT_CLASSES]
    )
    if lang == "zh":
        therapy = therapy.rename(columns={"Class": "类别", "Name": "名称", "Reference": "科研参考建议"})
    st.dataframe(therapy, use_container_width=True, hide_index=True)
    st.caption(t(lang, "disclaimer"))
