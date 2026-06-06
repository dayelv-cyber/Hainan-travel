from __future__ import annotations

import html
import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go

sys.path.append(str(Path(__file__).resolve().parent))

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
    DEFAULT_DATA_DIR,
    DEFAULT_MEAN_FILE,
    DEFAULT_NNLS_RESULT,
    DEFAULT_PATIENT_FILE,
    DEFAULT_PCA_PATIENT_IMAGE,
    DEFAULT_PCA_PURE_IMAGE,
    DEFAULT_PURE_FILE,
    DEFAULT_VALIDATION_FILE,
    THERAPY_NOTES_CN,
    THERAPY_NOTES_EN,
)
from core.export import export_workbook
from core.v2_analysis import run_v2_analysis


def tr(lang: str, zh: str, en: str) -> str:
    return zh if lang == "zh" else en


def labels_for(lang: str) -> dict[str, str]:
    return CLASS_LABELS_CN if lang == "zh" else CLASS_LABELS_EN


def class_name(cls: str, lang: str) -> str:
    labels = labels_for(lang)
    return f"{cls} {labels.get(cls, '')}".strip()


CLASS_ORDER = ["APA", "CAOX", "DCP", "DCY", "URIC"]
CLASS_COLORS = {
    "APA": "#1f77b4",
    "CAOX": "#ff7f0e",
    "DCP": "#2ca02c",
    "DCY": "#d62728",
    "URIC": "#9467bd",
}


def localize_class_columns(df: pd.DataFrame, lang: str) -> pd.DataFrame:
    out = df.copy()
    for col in ["NNLS主成分"]:
        if col in out.columns:
            out[col] = out[col].map(lambda x: class_name(str(x), lang) if pd.notna(x) else x)
    if "NNLS前2位" in out.columns:
        out = out.drop(columns=["NNLS前2位"])
    return out


def class_count_frame(summary: pd.DataFrame, lang: str) -> pd.DataFrame:
    counts = summary["NNLS主成分"].value_counts().reindex(CLASS_ORDER, fill_value=0).reset_index()
    counts.columns = ["Class", "Samples"]
    counts["Display"] = counts["Class"].map(lambda x: class_name(str(x), lang))
    return counts


def colored_bar_figure(df: pd.DataFrame, x: str, y: str, color_col: str, y_title: str, height: int = 260) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=df[x],
            y=df[y],
            marker_color=[CLASS_COLORS.get(str(cls), "#64748b") for cls in df[color_col]],
            text=df[y],
            textposition="outside",
            hovertemplate="%{x}<br>%{y}<extra></extra>",
        )
    )
    fig.update_layout(
        height=height,
        margin=dict(l=10, r=10, t=20, b=10),
        xaxis_title="",
        yaxis_title=y_title,
        showlegend=False,
    )
    fig.update_xaxes(tickangle=0, categoryorder="array", categoryarray=list(df[x]))
    return fig


def component_mapping_text(lang: str) -> str:
    labels = labels_for(lang)
    if lang == "zh":
        return "成分缩写对应关系： " + "； ".join(f"{cls}={labels.get(cls, '')}" for cls in DEFAULT_CLASSES)
    return "Component codes: " + "; ".join(f"{cls}={labels.get(cls, '')}" for cls in DEFAULT_CLASSES)


def summary_html_table(df: pd.DataFrame, lang: str) -> str:
    headers = [
        tr(lang, "样品", "Sample"),
        tr(lang, "NNLS主成分", "NNLS Top"),
        tr(lang, "主成分相对贡献", "Top Contribution"),
        tr(lang, "NNLS成分构成", "NNLS Composition"),
        tr(lang, "构成状态", "Composition Status"),
        tr(lang, "拟合残差", "Fitting Residual"),
        tr(lang, "残差标记", "Residual Flag"),
        tr(lang, "综合判断", "Summary"),
    ]
    rows = []
    for _, row in df.iterrows():
        sample = str(row["样品"])
        cells = [
            f'<a class="sample-link" href="?view=sample&sample={html.escape(sample)}">{html.escape(sample)}</a>',
            html.escape(class_name(str(row["NNLS主成分"]), lang)),
            f'{float(row["NNLS置信度"]):.4f}',
            html.escape(str(row["NNLS成分构成"])),
            html.escape(str(row["构成状态"])),
            f'{float(row["拟合残差"]):.4f}' if "拟合残差" in row else "",
            html.escape(str(row.get("残差标记", ""))),
            html.escape(str(row["综合判断"])),
        ]
        rows.append("<tr>" + "".join(f"<td>{cell}</td>" for cell in cells) + "</tr>")
    return (
        '<div class="table-wrap"><table class="sample-table">'
        + "<thead><tr>"
        + "".join(f"<th>{html.escape(h)}</th>" for h in headers)
        + "</tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table></div>"
    )


def uploaded_or_default(uploaded, default_path: Path):
    return uploaded if uploaded is not None else default_path


def source_name(uploaded, default_path: Path) -> str:
    return uploaded.name if uploaded is not None else default_path.name


def source_signature(uploaded, default_path: Path) -> tuple[str, int]:
    if uploaded is not None:
        return uploaded.name, int(getattr(uploaded, "size", 0) or 0)
    try:
        return default_path.name, int(default_path.stat().st_mtime)
    except OSError:
        return default_path.name, 0


LITERATURE_REFERENCES = [
    {
        "id": "[1]",
        "title": "Real-time and accurate calibration detection of gout stones based on terahertz and Raman spectroscopy",
        "use": "支持 THz、Raman、FTIR 对痛风/尿酸相关结石进行互补检测,说明 THz 可提供分子振动与组成信息。",
    },
    {
        "id": "[2]",
        "title": "Fast Detection of Uric Acid in Urine for Early Diagnosis Using THz Polarized Waves",
        "use": "支持尿酸相关物质在 THz 波段具有检测潜力,为 URIC 类别解释提供依据。",
    },
    {
        "id": "[3]",
        "title": "Terahertz Enhanced Sensing of Uric Acid Based on Metallic Slot Array Metamaterial",
        "use": "说明太赫兹增强传感可用于尿酸检测,但本系统未使用超材料增强,仅作为尿酸 THz 检测可行性参考。",
    },
    {
        "id": "[4]",
        "title": "基于太赫兹时域光谱技术的肾结石成分检测研究",
        "use": "本地中文论文,用于支撑 THz-TDS 在肾/尿路结石成分检测中的应用背景。",
    },
    {
        "id": "[5]",
        "title": "太赫兹频谱在尿路结石和咖啡识别中的应用研究",
        "use": "本地中文论文,用于支撑 THz 频谱结合模式识别/分类分析的研究背景。",
    },
    {
        "id": "[6]",
        "title": "Portable Point-of-Care Uric Acid Detection System with Cloud-Based Data Analysis and Patient Monitoring",
        "use": "支持检测结果可与云端分析和患者监测结合,为后续手机端/云端展示提供背景参考。",
    },
    {
        "id": "[7]",
        "title": "Urinary Metabolic Biomarker Profiling for Cancer Diagnosis by Terahertz Spectroscopy: Review and Perspective",
        "use": "综述性背景,说明 THz 技术在尿液/生物标志物分析中的潜力,不直接作为结石分类依据。",
    },
]


def research_basis(lang: str) -> pd.DataFrame:
    if lang == "zh":
        rows = [
            {
                "模块": "算法定位",
                "科研参考说明": "NNLS 将未知样品谱表示为多个纯品谱的非负线性组合,更符合混合结石“多成分叠加”的实际情况。输出应理解为光谱贡献比例,不是临床化学定量百分比。",
                "引用": "[1], [4], [5]",
            },
            {
                "模块": "拟合残差质控",
                "科研参考说明": "拟合残差衡量纯品基底线性组合对样品谱的解释程度。残差偏高说明当前纯品库不能充分解释该样品谱形,可能存在噪声、基线差异、混合复杂或基底外成分,应作为复核线索。",
                "引用": "[4], [5]",
            },
            {
                "模块": "URIC 尿酸",
                "科研参考说明": "尿酸及痛风石相关研究显示,THz/Raman/FTIR 等谱学方法可反映分子结构和组成差异。本系统中 URIC 高贡献提示尿酸相关谱型,仍需结合送检或临床结果确认。",
                "引用": "[1], [2], [3]",
            },
            {
                "模块": "CAOX 草酸钙",
                "科研参考说明": "草酸钙是尿路结石常见成分。本系统根据纯品草酸钙谱图进行谱型归属,CAOX 高贡献表示与草酸钙标准谱更接近。",
                "引用": "[4], [5]",
            },
            {
                "模块": "APA/DCP 磷酸盐类",
                "科研参考说明": "APA 与 DCP 都属于磷酸盐相关成分,谱形可能与混合结石、基线和其他磷酸盐类成分发生重叠。当前 APA/DCP 纯品数量较少,结果更应结合验证 tab 判断。",
                "引用": "[4], [5]",
            },
            {
                "模块": "云端/移动端扩展",
                "科研参考说明": "尿酸检测与患者监测已有便携化、云端化研究。本系统后续可把网页端报告和患者端小程序结合,但患者端只展示脱敏结果和科普解释。",
                "引用": "[6]",
            },
            {
                "模块": "局限依据",
                "科研参考说明": "文献和送检目录都表明尿路结石/痛风石可包含混合成分或基底外成分。当前系统以五类纯品基底和 NNLS 分解为核心,结果需要结合送检弱标签、谱图形态和残差标记综合解释。",
                "引用": "[1], [4], [5]",
            },
        ]
    else:
        rows = [
            {
                "Module": "Algorithm",
                "Research note": "NNLS represents an unknown spectrum as a non-negative combination of pure references, which fits the mixed-stone setting. The output is a spectral contribution estimate, not a clinical chemical percentage.",
                "Refs": "[1], [4], [5]",
            },
            {
                "Module": "Residual quality control",
                "Research note": "The fitting residual measures how well the current pure-reference basis explains a sample spectrum. A high residual flags noise, baseline mismatch, complex mixture, or out-of-basis components.",
                "Refs": "[4], [5]",
            },
            {
                "Module": "URIC",
                "Research note": "Studies on uric acid/gout stones support the use of THz/Raman/FTIR spectroscopy for molecular and compositional information.",
                "Refs": "[1], [2], [3]",
            },
            {
                "Module": "Cloud/mobile",
                "Research note": "Portable uric-acid detection with cloud analysis supports the later patient-side display concept.",
                "Refs": "[6]",
            },
            {
                "Module": "Limitations",
                "Research note": "Urinary/gout stones can be mixed and may contain out-of-basis components. The system should be interpreted with NNLS, weak-label validation, spectral morphology, and residual flags together.",
                "Refs": "[1], [4], [5]",
            },
        ]
    return pd.DataFrame(rows)


st.set_page_config(page_title=APP_NAME_CN, layout="wide")

st.markdown(
    """
    <style>
    .stApp {
        background:
            linear-gradient(135deg, rgba(178,34,34,.06) 0%, rgba(255,255,255,1) 42%, rgba(30,41,59,.05) 100%);
    }
    .hero {
        padding: 22px 26px;
        border-left: 6px solid #B22222;
        background: #fff;
        border-radius: 8px;
        box-shadow: 0 12px 28px rgba(15,23,42,.08);
    }
    .hero h1 { margin: 0 0 8px 0; letter-spacing: 0; }
    .hero p { margin: 0; color: #475569; }
    .note {
        padding: 12px 14px;
        border-radius: 8px;
        background: #fff7ed;
        border: 1px solid #fed7aa;
        color: #7c2d12;
    }
    .limit {
        padding: 12px 14px;
        border-radius: 8px;
        background: #fef2f2;
        border: 1px solid #fecaca;
        color: #7f1d1d;
    }
    .table-wrap {
        max-height: 520px;
        overflow: auto;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        background: #fff;
    }
    .sample-table {
        width: 100%;
        border-collapse: collapse;
        font-size: 14px;
    }
    .sample-table th {
        position: sticky;
        top: 0;
        background: #f8fafc;
        color: #64748b;
        font-weight: 600;
        border-bottom: 1px solid #e2e8f0;
        padding: 10px 12px;
        text-align: left;
        white-space: nowrap;
    }
    .sample-table td {
        border-bottom: 1px solid #e2e8f0;
        padding: 10px 12px;
        vertical-align: top;
        white-space: nowrap;
    }
    .sample-link {
        color: #b22222;
        font-weight: 700;
        text-decoration: none;
    }
    .sample-link:hover { text-decoration: underline; }
    </style>
    """,
    unsafe_allow_html=True,
)

lang_label = st.sidebar.radio("Language / 语言", ["中文", "English"], horizontal=True)
lang = "zh" if lang_label == "中文" else "en"

title = APP_NAME_CN if lang == "zh" else APP_NAME_EN
subtitle = tr(
    lang,
    "基于标准品谱图的尿路结石成分分析与科研参考系统",
    "Reference-spectrum-based urinary stone composition analysis system",
)
st.markdown(f"<div class='hero'><h1>{title}</h1><p>{subtitle}</p></div>", unsafe_allow_html=True)
st.warning(
    tr(
        lang,
        "以下结果仅供科研/教学参考，不作为临床诊断或治疗依据。概率不是严格化学定量百分比，应理解为基于纯品谱图的相似贡献比例。",
        "Results are for research/teaching only, not for clinical diagnosis or treatment. Probabilities are spectral contribution estimates, not strict chemical percentages.",
    )
)

st.sidebar.header(tr(lang, "数据与运行", "Data & Run"))
st.sidebar.caption(tr(lang, "默认读取最终文件/大创中的最新数据；也可以上传文件临时替换。", "Default data are read from 最终文件/大创; uploaded files can temporarily override them."))
st.sidebar.code(str(DEFAULT_DATA_DIR), language="text")

with st.sidebar.expander(tr(lang, "上传检测文件", "Upload Detection File"), expanded=True):
    st.caption(
        tr(
            lang,
            "上传一个原始重复谱 CSV 即可完成检测；后端会自动求平均、NNLS分解、拟合残差评估和送检验证。正常演示也可以不上传，系统会读取默认文件。",
            "Upload one raw replicate-spectra CSV; backend will average replicates, run NNLS, residual quality control, and validation. For demos, uploading is optional.",
        )
    )
    raw_upload = st.file_uploader(
        tr(lang, "原始重复谱 CSV", "Raw replicate spectra CSV"),
        type=["csv"],
        help=tr(lang, "格式示例：第一列 wave_num，后续列为 S31-1、S31-2 等重复谱。", "Format: first column wave_num, followed by replicate columns such as S31-1, S31-2."),
    )

active_sources = {
    "raw": source_name(raw_upload, DEFAULT_PATIENT_FILE),
}
st.sidebar.caption(
    tr(
        lang,
        "当前检测文件: " + active_sources["raw"],
        "Current detection file: " + active_sources["raw"],
    )
)
run = st.sidebar.button(tr(lang, "运行/刷新分析", "Run / Refresh"), type="primary")

input_signature = (
    source_signature(raw_upload, DEFAULT_PATIENT_FILE),
)
if st.session_state.get("input_signature") != input_signature:
    st.session_state.pop("v2_analysis", None)
    st.session_state.input_signature = input_signature

if "v2_analysis" not in st.session_state or run:
    with st.spinner(tr(lang, "正在计算 NNLS 分解、拟合残差和验证指标...", "Running NNLS, residual quality control, and validation...")):
        st.session_state.v2_analysis = run_v2_analysis(
            pure_path=DEFAULT_PURE_FILE,
            mean_path=DEFAULT_MEAN_FILE,
            raw_path=DEFAULT_PATIENT_FILE,
            validation_path=DEFAULT_VALIDATION_FILE,
            uploaded_raw_source=raw_upload,
        )

analysis = st.session_state.v2_analysis
summary = analysis["summary"]
validation = analysis["validation"]
metrics = analysis["validation_metrics"]
component_accuracy = analysis["component_accuracy"]
classes = analysis["classes"]

metric_cols = st.columns(5)
metric_cols[0].metric(tr(lang, "样品数", "Samples"), len(summary))
metric_cols[1].metric(tr(lang, "类别数", "Classes"), len(classes))
metric_cols[2].metric(tr(lang, "合理归属", "Reasonable"), f"{metrics['reasonable']}/{metrics['total']}")
metric_cols[3].metric(tr(lang, "组合命中", "Top-2"), f"{metrics['combo']}/{metrics['total']}")
high_residual_count = int((summary["残差标记"] == "拟合残差偏高").sum()) if "残差标记" in summary.columns else 0
metric_cols[4].metric(tr(lang, "残差复核", "Residual Review"), high_residual_count)

count_df = class_count_frame(summary, lang)
top_cls = count_df.sort_values("Samples", ascending=False).iloc[0]
st.markdown("### " + tr(lang, "本批样品结论概览", "Batch Overview"))
st.plotly_chart(
    colored_bar_figure(count_df, x="Display", y="Samples", color_col="Class", y_title=tr(lang, "样品数", "Samples"), height=240),
    use_container_width=True,
)
st.caption(
    tr(
        lang,
        f"本批 {len(summary)} 例样品中，{class_name(top_cls['Class'], lang)} 作为 NNLS 最高贡献成分的样品最多，共 {int(top_cls['Samples'])} 例。计数基于最高贡献成分，不代表真实临床构成比例。",
        f"In this batch of {len(summary)} samples, {class_name(top_cls['Class'], lang)} appears most often as the top NNLS contributor ({int(top_cls['Samples'])} cases). Counts are based on top contribution, not clinical composition percentages.",
    )
)

st.markdown(
    "<div class='note'>"
    + component_mapping_text(lang)
    + "</div>",
    unsafe_allow_html=True,
)

def query_value(name: str, default: str = "") -> str:
    value = st.query_params.get(name, default)
    if isinstance(value, list):
        return value[0] if value else default
    return str(value)


views = [
    ("main", tr(lang, "NNLS主表", "NNLS Table")),
    ("validation", tr(lang, "验证", "Validation")),
    ("sample", tr(lang, "样品详情", "Sample Detail")),
    ("pca", tr(lang, "PCA 图", "PCA Figures")),
    ("reference", tr(lang, "科研参考", "Research Notes")),
]
view_keys = [item[0] for item in views]
view_labels = [item[1] for item in views]
current_view = query_value("view", "main")
if current_view not in view_keys:
    current_view = "main"
selected_label = st.radio(
    tr(lang, "页面", "Page"),
    view_labels,
    index=view_keys.index(current_view),
    horizontal=True,
    label_visibility="collapsed",
)
view = view_keys[view_labels.index(selected_label)]
if view != current_view:
    st.query_params["view"] = view

if view == "main":
    main_cols = [
        "样品",
        "NNLS主成分",
        "NNLS置信度",
        "NNLS成分构成",
        "构成状态",
        "拟合残差",
        "残差标记",
        "综合判断",
    ]
    st.caption(tr(lang, "点击样品编号可进入样品详情。", "Click a sample ID to open its detail page."))
    st.markdown(summary_html_table(summary[main_cols], lang), unsafe_allow_html=True)
    if st.button(tr(lang, "导出 Excel", "Export Excel")):
        path = export_workbook(
            {
                "NNLS主表": summary.drop(columns=["NNLS前2位", "S69提示"], errors="ignore"),
                "验证": validation.drop(columns=["S69提示"], errors="ignore"),
                "成分级准确率": component_accuracy,
            },
            prefix="尿结石NNLS_验证结果",
        )
        st.success(str(path))

elif view == "validation":
    st.info(
        tr(
            lang,
            "验证使用送检弱标签：上理工粗分类 + 长海详细成分。界面仅使用脱敏列：样品号、粗标签、详细成分。",
            "Validation uses weak labels: coarse category and detailed composition. Only de-identified fields are used.",
        )
    )
    vcols = st.columns(3)
    vcols[0].metric(tr(lang, "严格口径", "Strict"), f"{metrics['strict']}/{metrics['total']}")
    vcols[0].caption(tr(lang, "NNLS主成分 = 上理工粗标签，对混合结石偏严。", "Top NNLS equals coarse label; strict for mixed stones."))
    vcols[1].metric(tr(lang, "合理归属", "Reasonable"), f"{metrics['reasonable']}/{metrics['total']}")
    vcols[1].caption(tr(lang, "NNLS主成分包含在长海详细成分中，作为主指标。", "Top NNLS appears in detailed composition; primary metric."))
    vcols[2].metric(tr(lang, "成分组合", "Top-2 Combo"), f"{metrics['combo']}/{metrics['total']}")
    vcols[2].caption(tr(lang, "NNLS前二成分与详细成分有交集，更贴合混合本质。", "Top-2 NNLS overlaps detailed composition."))
    st.subheader(tr(lang, "成分预测准确率", "Component-Level Accuracy"))
    comp_display = component_accuracy.copy()
    comp_display["成分"] = comp_display["成分"].map(lambda x: class_name(str(x), lang))
    st.dataframe(comp_display, use_container_width=True, hide_index=True)
    st.caption(
        tr(
            lang,
            "检出率表示送检中出现该成分时，NNLS前二成分是否覆盖它；预测准确率表示NNLS前二成分预测到该类时，该类是否出现在送检详细成分中。",
            "Recall shows whether top-2 NNLS covers a component present in validation; precision shows whether a predicted top-2 component appears in the detailed validation composition.",
        )
    )
    st.dataframe(localize_class_columns(validation.drop(columns=["S69提示"], errors="ignore"), lang), use_container_width=True, hide_index=True)

elif view == "sample":
    sample_list = list(summary["样品"])
    query_sample = query_value("sample", sample_list[0] if sample_list else "")
    sample_index = sample_list.index(query_sample) if query_sample in sample_list else 0
    sample = st.selectbox(tr(lang, "选择样品", "Select sample"), sample_list, index=sample_index)
    row = summary.loc[summary["样品"] == sample].iloc[0]
    left, right = st.columns([1, 1])
    with left:
        st.subheader(tr(lang, "NNLS 五成分概率", "NNLS Probabilities"))
        prob_df = pd.DataFrame(
            {
                "Class": classes,
                tr(lang, "成分", "Class"): [class_name(cls, lang) for cls in classes],
                tr(lang, "概率", "Probability"): [float(row[cls]) for cls in classes],
            }
        )
        st.plotly_chart(
            colored_bar_figure(
                prob_df,
                x=tr(lang, "成分", "Class"),
                y=tr(lang, "概率", "Probability"),
                color_col="Class",
                y_title=tr(lang, "相对贡献", "Relative Contribution"),
                height=300,
            ),
            use_container_width=True,
        )
        st.dataframe(prob_df.drop(columns=["Class"]), use_container_width=True, hide_index=True)
        if "拟合残差" in row:
            st.metric(tr(lang, "拟合残差", "Fitting Residual"), f"{float(row['拟合残差']):.4f}", str(row.get("残差标记", "")))
    with right:
        st.subheader(tr(lang, "平均谱", "Mean Spectrum"))
        mean_df = analysis["mean_df"]
        if sample in mean_df.columns:
            spectrum = mean_df[["wave_num", sample]].rename(columns={"wave_num": "THz", sample: tr(lang, "强度", "Intensity")})
            st.line_chart(spectrum, x="THz", y=tr(lang, "强度", "Intensity"), height=300)
        else:
            st.warning(tr(lang, "未找到该样品的平均谱。", "Mean spectrum not found."))

elif view == "pca":
    c1, c2 = st.columns(2)
    if raw_upload is None:
        st.caption(
            tr(
                lang,
                "默认模式显示由 PCA.py 生成的标准 PCA 图；上传临时检测文件时，会按同一 PCA 流程动态投影当前样品。",
                "Default mode shows the PCA.py-generated reference figures; uploads are projected dynamically with the same PCA workflow.",
            )
        )
        with c1:
            st.subheader(tr(lang, "纯品 PCA 分类", "Pure PCA"))
            if DEFAULT_PCA_PURE_IMAGE.exists():
                st.image(str(DEFAULT_PCA_PURE_IMAGE), use_container_width=True)
            else:
                st.warning(str(DEFAULT_PCA_PURE_IMAGE))
        with c2:
            st.subheader(tr(lang, "病人样本 PCA 投影", "Patient Projection"))
            if DEFAULT_PCA_PATIENT_IMAGE.exists():
                st.image(str(DEFAULT_PCA_PATIENT_IMAGE), use_container_width=True)
            else:
                st.warning(str(DEFAULT_PCA_PATIENT_IMAGE))
    else:
        st.caption(
            tr(
                lang,
                "当前为临时上传文件投影图：纯品 PCA 方向、配色与原 PCA.py 保持一致，当前样品以黑色 × 标出。",
                "Current upload projection: PCA orientation and colors follow PCA.py; current samples are marked as black x.",
            )
        )
        pca_pure = analysis["pca_pure"].copy()
        pca_sample = analysis["pca_sample"].copy()

        def add_pure_traces(fig):
            for cls in CLASS_ORDER:
                sub = pca_pure[pca_pure["类别"] == cls]
                if sub.empty:
                    continue
                fig.add_trace(
                    go.Scatter(
                        x=sub["PC1"],
                        y=sub["PC2"],
                        mode="markers",
                        name=class_name(cls, lang),
                        text=sub["名称"],
                        marker=dict(color=CLASS_COLORS.get(cls, "#777"), size=8, opacity=0.75),
                        hovertemplate="%{text}<br>PC1=%{x:.2f}<br>PC2=%{y:.2f}<extra></extra>",
                    )
                )

        with c1:
            st.subheader(tr(lang, "纯品 PCA 分类", "Pure PCA"))
            fig_pure = go.Figure()
            add_pure_traces(fig_pure)
            fig_pure.update_layout(xaxis_title="PC1", yaxis_title="PC2", height=420, margin=dict(l=8, r=8, t=16, b=8))
            st.plotly_chart(fig_pure, use_container_width=True)
        with c2:
            st.subheader(tr(lang, "病人样本 PCA 投影", "Patient Projection"))
            fig_patient = go.Figure()
            add_pure_traces(fig_patient)
            fig_patient.add_trace(
                go.Scatter(
                    x=pca_sample["PC1"],
                    y=pca_sample["PC2"],
                    mode="markers",
                    name=tr(lang, "当前样品", "Current samples"),
                    text=pca_sample["名称"],
                    marker=dict(color="black", size=11, symbol="x", line=dict(width=2)),
                    hovertemplate="%{text}<br>PC1=%{x:.2f}<br>PC2=%{y:.2f}<extra></extra>",
                )
            )
            fig_patient.update_layout(xaxis_title="PC1", yaxis_title="PC2", height=420, margin=dict(l=8, r=8, t=16, b=8))
            st.plotly_chart(fig_patient, use_container_width=True)
        st.caption(
            tr(
                lang,
                f"当前 PCA 由纯品数据拟合，并投影当前 {len(pca_sample)} 个样品；前两主成分累计解释率 {analysis['pca_explained_variance']:.2f}。",
                f"PCA is fitted on pure spectra and projects {len(pca_sample)} current samples; PC1+PC2 explained variance is {analysis['pca_explained_variance']:.2f}.",
            )
        )

elif view == "reference":
    label_map = labels_for(lang)
    notes = THERAPY_NOTES_CN if lang == "zh" else THERAPY_NOTES_EN
    if "show_reference_tables" not in st.session_state:
        st.session_state.show_reference_tables = False
    if st.button(tr(lang, "方法依据和参考文献", "Method Basis and References"), type="primary"):
        st.session_state.show_reference_tables = not st.session_state.show_reference_tables
    st.caption(
        tr(
            lang,
            "点击按钮展开或收起方法依据、成分说明与参考文献。",
            "Click the button to show or hide method basis, component notes, and references.",
        )
    )
    if st.session_state.show_reference_tables:
        st.subheader(tr(lang, "成分解释", "Component Notes"))
        ref = pd.DataFrame(
            {
                tr(lang, "类别", "Class"): [class_name(cls, lang) for cls in DEFAULT_CLASSES],
                tr(lang, "科研参考说明", "Research note"): [notes.get(cls, "") for cls in DEFAULT_CLASSES],
            }
        )
        st.dataframe(ref, use_container_width=True, hide_index=True)

        st.subheader(tr(lang, "方法与应用依据", "Method and Application Basis"))
        st.dataframe(research_basis(lang), use_container_width=True, hide_index=True)

        st.subheader(tr(lang, "参考文献", "References"))
        ref_rows = []
        for item in LITERATURE_REFERENCES:
            if lang == "zh":
                ref_rows.append({"编号": item["id"], "文献": item["title"], "本系统参考用途": item["use"]})
            else:
                ref_rows.append({"No.": item["id"], "Reference": item["title"], "Use in this system": item["use"]})
        st.dataframe(pd.DataFrame(ref_rows), use_container_width=True, hide_index=True)
        st.caption(
            tr(
                lang,
                "注：参考文献用于说明太赫兹/拉曼/红外谱学在尿酸、痛风石、尿路结石及生物样本检测中的研究背景。本系统输出仍以当前纯品库、NNLS 分解、拟合残差质控和送检弱标签验证为准。",
                "Note: references provide background for THz/Raman/FTIR spectroscopy in uric acid, gout stones, urinary stones, and biological detection. System outputs depend on the current reference spectra, NNLS, residual quality control, and weak-label validation.",
            )
        )
        st.caption(
            tr(
                lang,
                "以上说明用于科研展示和报告解释，不构成诊断或治疗建议。",
                "These notes are for research presentation only and are not diagnosis or treatment advice.",
            )
        )
