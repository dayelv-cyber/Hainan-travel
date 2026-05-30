from pathlib import Path


APP_NAME_CN = "尿结石 FTIR 标准品比对与自动初筛系统"
APP_NAME_EN = "FTIR-Based Urinary Stone Screening System"

PROJECT_DIR = Path(__file__).resolve().parent
WORKSPACE_DIR = PROJECT_DIR.parent
DEFAULT_DATA_DIR = WORKSPACE_DIR / "大创2" / "大创"
DEFAULT_PURE_FILE = DEFAULT_DATA_DIR / "data-纯品.csv"
DEFAULT_PATIENT_FILE = DEFAULT_DATA_DIR / "data-原始.csv"
DEFAULT_EXISTING_RESULT = DEFAULT_DATA_DIR / "病人结石混合谱型分析.csv"
OUTPUT_DIR = PROJECT_DIR / "outputs"

FREQ_MIN = 1.0
FREQ_MAX = 4.0
SG_WINDOW = 11
SG_POLY = 3
SG_DERIV = 2
PCA_COMPONENTS = 8

CM_TO_THZ = 1 / 33.356
MINOR_PROB_THRESHOLD = 0.25
PEAK_TOLERANCE = 0.10

DEFAULT_CLASSES = ["COM", "UA", "MAP", "CYS"]
CLASS_LABELS_CN = {
    "COM": "草酸钙",
    "UA": "尿酸",
    "MAP": "磷酸铵镁",
    "CYS": "胱氨酸",
    "Unknown": "未知",
}
CLASS_LABELS_EN = {
    "COM": "Calcium oxalate",
    "UA": "Uric acid",
    "MAP": "Magnesium ammonium phosphate",
    "CYS": "Cystine",
    "Unknown": "Unknown",
}

PEAK_RULES = {
    "COM": {"peaks": [3.00, 3.03], "status": "primary"},
    "UA": {"peaks": [1.42, 2.39, 2.92, 3.98], "status": "primary"},
    "CYS": {"peaks": [1.49, 2.09, 2.96], "status": "reference_pending"},
    "MAP": {"peaks": [1.97, 2.85, 3.41], "status": "reference_pending"},
}

THERAPY_NOTES_CN = {
    "COM": "限草酸饮食、增加饮水；必要时由医生评估药物预防。",
    "UA": "低嘌呤饮食、碱化尿液、控制尿酸；需结合临床检查。",
    "MAP": "提示可能与泌尿系感染相关，建议进一步感染相关检查。",
    "CYS": "提示可能与胱氨酸尿症相关，需长期管理和专科评估。",
    "Unknown": "结果不确定，建议结合标准成分分析进一步复核。",
}
THERAPY_NOTES_EN = {
    "COM": "Reduce oxalate load, increase hydration, and consider medical prevention under clinical guidance.",
    "UA": "Consider low-purine diet, urine alkalization, and uric acid control with clinical confirmation.",
    "MAP": "May indicate infection-related stones; further infection workup is recommended.",
    "CYS": "May indicate cystinuria-related stones; long-term specialist management is needed.",
    "Unknown": "Uncertain result; confirm with standard compositional analysis.",
}

