TEXT = {
    "zh": {
        "title": "尿结石 FTIR 标准品比对与自动初筛系统",
        "subtitle": "基于标准品谱图的自动谱型归属工具，仅供科研参考",
        "upload": "上传合并 CSV",
        "demo": "使用学长 data-原始.csv 演示",
        "mode": "分析模式",
        "run": "运行分析",
        "summary": "样品级汇总",
        "details": "重复谱明细",
        "spectrum": "光谱曲线",
        "similarity": "纯品相似度",
        "rules": "峰位规则",
        "export": "导出 Excel",
        "sample": "选择样品",
        "disclaimer": "以下结果仅供科研/教学参考，不作为临床诊断或治疗依据。",
    },
    "en": {
        "title": "FTIR-Based Urinary Stone Screening System",
        "subtitle": "Standard-reference spectral typing tool for research use only",
        "upload": "Upload merged CSV",
        "demo": "Use demo data-原始.csv",
        "mode": "Analysis mode",
        "run": "Run analysis",
        "summary": "Sample summary",
        "details": "Replicate details",
        "spectrum": "Spectra",
        "similarity": "Pure reference similarity",
        "rules": "Peak rules",
        "export": "Export Excel",
        "sample": "Select sample",
        "disclaimer": "Results are for research/teaching only and are not clinical diagnosis or treatment advice.",
    },
}


def t(lang: str, key: str) -> str:
    return TEXT.get(lang, TEXT["zh"]).get(key, key)

