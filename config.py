from pathlib import Path


APP_NAME_CN = "尿结石 FTIR 标准品比对与自动初筛系统"
APP_NAME_EN = "FTIR-Based Urinary Stone Screening System"

PROJECT_DIR = Path(__file__).resolve().parent
WORKSPACE_DIR = PROJECT_DIR.parent
DEFAULT_DATA_DIR = WORKSPACE_DIR / "最终文件" / "大创"
DEFAULT_PURE_FILE = DEFAULT_DATA_DIR / "data-纯品.csv"
DEFAULT_PATIENT_FILE = DEFAULT_DATA_DIR / "data-原始.csv"
DEFAULT_MEAN_FILE = DEFAULT_DATA_DIR / "data-均值.csv"
DEFAULT_NNLS_RESULT = DEFAULT_DATA_DIR / "混合物分解概率结果.csv"
DEFAULT_VALIDATION_FILE = DEFAULT_DATA_DIR / "送检目录(2).xlsx"
DEFAULT_PCA_PURE_IMAGE = DEFAULT_DATA_DIR / "纯品PCA分类.png"
DEFAULT_PCA_PATIENT_IMAGE = DEFAULT_DATA_DIR / "病人结石PCA投影.png"
DEFAULT_EXISTING_RESULT = DEFAULT_NNLS_RESULT
OUTPUT_DIR = PROJECT_DIR / "outputs"

FREQ_MIN = 1.0
FREQ_MAX = 4.0
SG_WINDOW = 11
SG_POLY = 3
SG_DERIV = 2
PCA_COMPONENTS = 8

CM_TO_THZ = 1 / 33.356
MINOR_PROB_THRESHOLD = 0.10
PEAK_TOLERANCE = 0.10

DEFAULT_CLASSES = ["CAOX", "DCY", "URIC", "APA", "DCP"]
CLASS_LABELS_CN = {
    "CAOX": "草酸钙",
    "DCY": "胱氨酸",
    "URIC": "尿酸",
    "APA": "磷灰石/碳酸磷灰石",
    "DCP": "磷酸氢钙",
    "AMP": "磷酸铵镁",
    "Unknown": "未知",
}
CLASS_LABELS_EN = {
    "CAOX": "Calcium oxalate",
    "DCY": "Cystine",
    "URIC": "Uric acid",
    "APA": "Apatite / carbonate apatite",
    "DCP": "Dicalcium phosphate",
    "AMP": "Magnesium ammonium phosphate",
    "Unknown": "Unknown",
}

PEAK_RULES = {
    "CAOX": {"peaks": [3.00, 3.03], "status": "primary"},
    "URIC": {"peaks": [1.42, 2.39, 2.92, 3.98], "status": "primary"},
    "DCY": {"peaks": [1.49, 2.09, 2.96], "status": "reference_pending"},
    "APA": {"peaks": [], "status": "reference_pending"},
    "DCP": {"peaks": [], "status": "reference_pending"},
}

THERAPY_NOTES_CN = {
    "CAOX": "科研参考：草酸钙相关样本通常关注饮水、草酸摄入与复发风险管理，需结合临床确认。",
    "DCY": "科研参考：胱氨酸相关样本提示可能涉及代谢因素，需专科进一步评估。",
    "URIC": "科研参考：尿酸相关样本通常关注嘌呤代谢、尿液 pH 与尿酸水平，需结合临床检查。",
    "APA": "科研参考：磷灰石/碳酸磷灰石相关样本可与感染、尿液环境等因素有关，需进一步复核。",
    "DCP": "科研参考：磷酸氢钙相关样本需结合尿液理化指标和标准成分分析判断。",
    "Unknown": "结果不确定，建议结合标准成分分析进一步复核。",
}
THERAPY_NOTES_EN = {
    "CAOX": "Research note: calcium oxalate results should be interpreted with hydration, oxalate load, and recurrence risk under clinical confirmation.",
    "DCY": "Research note: cystine-related results may suggest metabolic factors and require specialist confirmation.",
    "URIC": "Research note: uric-acid-related results should be interpreted with purine metabolism, urine pH, and clinical tests.",
    "APA": "Research note: apatite/carbonate apatite may relate to infection or urine chemistry and requires confirmation.",
    "DCP": "Research note: dicalcium phosphate results should be interpreted with urine chemistry and standard compositional analysis.",
    "Unknown": "Uncertain result; confirm with standard compositional analysis.",
}
