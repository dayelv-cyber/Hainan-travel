from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

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


def localize_class_columns(df: pd.DataFrame, lang: str) -> pd.DataFrame:
    out = df.copy()
    for col in ["NNLS主成分", "SVM主判"]:
        if col in out.columns:
            out[col] = out[col].map(lambda x: class_name(str(x), lang) if pd.notna(x) else x)
    if "NNLS前2位" in out.columns:
        out = out.drop(columns=["NNLS前2位"])
    return out


def class_count_frame(summary: pd.DataFrame, lang: str) -> pd.DataFrame:
    counts = summary["NNLS主成分"].value_counts().reindex(DEFAULT_CLASSES, fill_value=0).reset_index()
    counts.columns = ["Class", "Samples"]
    counts["Display"] = counts["Class"].map(lambda x: class_name(str(x), lang))
    return counts


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
                "模块": "SVM 辅助",
                "科研参考说明": "SVM 提供单标签分类视角,适合与 NNLS 形成交叉验证。若两者一致,可信度提高;若不一致,常提示混合成分或基底外成分,需要复核。",
                "引用": "[5]",
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
                "科研参考说明": "文献和送检目录都表明尿路结石/痛风石可包含混合成分或基底外成分。当前纯品库未覆盖 AMP/MAP,且 APA/DCP 纯品数量较少,因此 S69 等样品必须标注为基底未覆盖或需复核。",
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
                "Module": "SVM",
                "Research note": "SVM provides a single-label classification view. Agreement with NNLS increases confidence, while disagreement flags mixed or out-of-basis samples.",
                "Refs": "[5]",
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
                "Research note": "Urinary/gout stones can be mixed and may contain out-of-basis components. AMP/MAP is not covered by the current reference set, and APA/DCP have limited pure spectra.",
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

with st.sidebar.expander(tr(lang, "高级选项：上传/替换文件", "Advanced: Upload / Override Files"), expanded=False):
    st.caption(
        tr(
            lang,
            "正常演示无需上传，系统会自动读取默认文件。这里仅用于临时替换算法结果、纯品库、样品谱或脱敏验证表。",
            "No upload is needed for normal demos. Use this area only to temporarily override results, references, sample spectra, or validation files.",
        )
    )
    nnls_upload = st.file_uploader(
        tr(lang, "NNLS概率结果 CSV", "NNLS result CSV"),
        type=["csv"],
        help=tr(lang, "默认使用 混合物分解概率结果.csv", "Default: 混合物分解概率结果.csv"),
    )
    pure_upload = st.file_uploader(
        tr(lang, "纯品数据 CSV", "Pure reference CSV"),
        type=["csv"],
        help=tr(lang, "用于训练 SVM。默认使用 data-纯品.csv", "Used for SVM training. Default: data-纯品.csv"),
    )
    mean_upload = st.file_uploader(
        tr(lang, "样品平均谱 CSV", "Sample mean spectra CSV"),
        type=["csv"],
        help=tr(lang, "用于 SVM 预测和样品详情曲线。默认使用 data-均值.csv", "Used for SVM prediction and sample curves. Default: data-均值.csv"),
    )
    validation_upload = st.file_uploader(
        tr(lang, "脱敏送检对照 Excel", "De-identified validation Excel"),
        type=["xlsx", "xls"],
        help=tr(lang, "需包含 Sheet1: 序号/上理工/长海。不要上传含姓名住院号的公开版本。", "Requires Sheet1 with sample/coarse/detail fields."),
    )

active_sources = {
    "nnls": source_name(nnls_upload, DEFAULT_NNLS_RESULT),
    "pure": source_name(pure_upload, DEFAULT_PURE_FILE),
    "mean": source_name(mean_upload, DEFAULT_MEAN_FILE),
    "validation": source_name(validation_upload, DEFAULT_VALIDATION_FILE),
}
st.sidebar.caption(
    tr(
        lang,
        "当前文件: "
        + f"NNLS={active_sources['nnls']}; 纯品={active_sources['pure']}; 平均谱={active_sources['mean']}; 验证={active_sources['validation']}",
        "Current files: "
        + f"NNLS={active_sources['nnls']}; pure={active_sources['pure']}; mean={active_sources['mean']}; validation={active_sources['validation']}",
    )
)
run = st.sidebar.button(tr(lang, "运行/刷新分析", "Run / Refresh"), type="primary")

input_signature = (
    source_signature(nnls_upload, DEFAULT_NNLS_RESULT),
    source_signature(pure_upload, DEFAULT_PURE_FILE),
    source_signature(mean_upload, DEFAULT_MEAN_FILE),
    source_signature(validation_upload, DEFAULT_VALIDATION_FILE),
)
if st.session_state.get("input_signature") != input_signature:
    st.session_state.pop("v2_analysis", None)
    st.session_state.input_signature = input_signature

if "v2_analysis" not in st.session_state or run:
    with st.spinner(tr(lang, "正在读取 NNLS 结果、训练 SVM 并计算验证指标...", "Loading NNLS, training SVM, and validating...")):
        st.session_state.v2_analysis = run_v2_analysis(
            nnls_path=uploaded_or_default(nnls_upload, DEFAULT_NNLS_RESULT),
            pure_path=uploaded_or_default(pure_upload, DEFAULT_PURE_FILE),
            mean_path=uploaded_or_default(mean_upload, DEFAULT_MEAN_FILE),
            raw_path=DEFAULT_PATIENT_FILE,
            validation_path=uploaded_or_default(validation_upload, DEFAULT_VALIDATION_FILE),
        )

analysis = st.session_state.v2_analysis
summary = analysis["summary"]
validation = analysis["validation"]
metrics = analysis["validation_metrics"]
classes = analysis["classes"]

metric_cols = st.columns(5)
metric_cols[0].metric(tr(lang, "样品数", "Samples"), len(summary))
metric_cols[1].metric(tr(lang, "类别数", "Classes"), len(classes))
metric_cols[2].metric(tr(lang, "合理归属", "Reasonable"), f"{metrics['reasonable']}/{metrics['total']}")
metric_cols[3].metric(tr(lang, "组合命中", "Top-2"), f"{metrics['combo']}/{metrics['total']}")
metric_cols[4].metric("SVM PCA", f"{analysis['svm_pca_explained_variance']:.2f}" if analysis["svm_pca_explained_variance"] else "-")

count_df = class_count_frame(summary, lang)
top_cls = count_df.sort_values("Samples", ascending=False).iloc[0]
st.markdown("### " + tr(lang, "本批样品结论概览", "Batch Overview"))
st.bar_chart(count_df, x="Display", y="Samples", height=230)
st.caption(
    tr(
        lang,
        f"本批 {len(summary)} 例样品中，{class_name(top_cls['Class'], lang)} 作为 NNLS 最高贡献成分的样品最多，共 {int(top_cls['Samples'])} 例。计数基于最高贡献成分，不代表真实临床构成比例。",
        f"In this batch of {len(summary)} samples, {class_name(top_cls['Class'], lang)} appears most often as the top NNLS contributor ({int(top_cls['Samples'])} cases). Counts are based on top contribution, not clinical composition percentages.",
    )
)

st.markdown(
    "<div class='note'>"
    + tr(
        lang,
        "两方法不一致往往指向混合结石：SVM 被迫单选一个主类，NNLS 则会显示多个成分的贡献。因此“不一致”是复核提示，不一定是程序错误。",
        "Disagreement can indicate mixed stones: SVM must choose one class, while NNLS can expose multiple spectral contributors.",
    )
    + "</div>",
    unsafe_allow_html=True,
)

tab_main, tab_validation, tab_sample, tab_pca, tab_reference = st.tabs(
    [
        tr(lang, "双方法主表", "Dual-Method Table"),
        tr(lang, "验证", "Validation"),
        tr(lang, "样品详情", "Sample Detail"),
        tr(lang, "PCA 图", "PCA Figures"),
        tr(lang, "科研参考", "Research Notes"),
    ]
)

with tab_main:
    display_cols = [
        "样品",
        "NNLS主成分",
        "NNLS置信度",
        "NNLS成分构成",
        "SVM主判",
        "SVM置信度",
        "双法是否一致",
        "综合判断",
    ]
    st.dataframe(localize_class_columns(summary[display_cols], lang), use_container_width=True, hide_index=True)
    if st.button(tr(lang, "导出 Excel", "Export Excel")):
        path = export_workbook(
            {
                "双方法主表": summary.drop(columns=["NNLS前2位", "S69提示"], errors="ignore"),
                "验证": validation.drop(columns=["S69提示"], errors="ignore"),
            },
            prefix="尿结石NNLS_SVM_验证结果",
        )
        st.success(str(path))

with tab_validation:
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
    st.dataframe(localize_class_columns(validation.drop(columns=["S69提示"], errors="ignore"), lang), use_container_width=True, hide_index=True)

with tab_sample:
    sample = st.selectbox(tr(lang, "选择样品", "Select sample"), list(summary["样品"]))
    row = summary.loc[summary["样品"] == sample].iloc[0]
    left, right = st.columns([1, 1])
    with left:
        st.subheader(tr(lang, "NNLS 五成分概率", "NNLS Probabilities"))
        prob_df = pd.DataFrame(
            {
                tr(lang, "成分", "Class"): [class_name(cls, lang) for cls in classes],
                tr(lang, "概率", "Probability"): [float(row[cls]) for cls in classes],
            }
        )
        st.bar_chart(prob_df, x=tr(lang, "成分", "Class"), y=tr(lang, "概率", "Probability"), height=300)
        st.dataframe(prob_df, use_container_width=True, hide_index=True)
    with right:
        st.subheader(tr(lang, "平均谱", "Mean Spectrum"))
        mean_df = analysis["mean_df"]
        if sample in mean_df.columns:
            spectrum = mean_df[["wave_num", sample]].rename(columns={"wave_num": "THz", sample: tr(lang, "强度", "Intensity")})
            st.line_chart(spectrum, x="THz", y=tr(lang, "强度", "Intensity"), height=300)
        else:
            st.warning(tr(lang, "未找到该样品的平均谱。", "Mean spectrum not found."))

with tab_pca:
    st.caption(tr(lang, "PCA 图由 PCA.py 生成；修改 NNLS 的数值校正不会改变 PCA 图。", "PCA figures are generated by PCA.py. NNLS numeric changes do not alter PCA figures."))
    c1, c2 = st.columns(2)
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

with tab_reference:
    label_map = labels_for(lang)
    notes = THERAPY_NOTES_CN if lang == "zh" else THERAPY_NOTES_EN
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
            "注：参考文献用于说明太赫兹/拉曼/红外谱学在尿酸、痛风石、尿路结石及生物样本检测中的研究背景。本系统输出仍以当前纯品库、NNLS 分解、SVM 辅助和送检弱标签验证为准。",
            "Note: references provide background for THz/Raman/FTIR spectroscopy in uric acid, gout stones, urinary stones, and biological detection. System outputs depend on the current reference spectra, NNLS, SVM, and weak-label validation.",
        )
    )
    st.caption(
        tr(
            lang,
            "以上说明用于科研展示和报告解释，不构成诊断或治疗建议。",
            "These notes are for research presentation only and are not diagnosis or treatment advice.",
        )
    )
