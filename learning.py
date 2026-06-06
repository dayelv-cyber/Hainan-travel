# =========================================================
# 尿路结石 THz 谱型识别系统（教学注释版，1-4 THz）
# =========================================================
#
# 本文件基于学长的“最终模型.py”整理而来，核心算法流程保持一致：
#
#   纯品数据 data-纯品.csv
#       -> 截取 1-4 THz
#       -> SNV 标准化
#       -> SG 二阶导数
#       -> StandardScaler
#       -> PCA 降维
#       -> SVM 分类器训练
#
#   病人样本 data-原始.csv
#       -> 使用同样的预处理方式
#       -> 使用纯品训练得到的 scaler / PCA / SVM
#       -> 输出 Predicted_Type 和各类别概率
#
# 重要定位：
#   1. 纯品数据有类别标签，所以用于训练模型。
#   2. 病人样本没有真实成分标签，所以这里只能做“谱型初筛/归属”，不能证明准确率。
#   3. 当前模型不是精确定量模型，输出概率更适合理解为“谱型相似倾向”，不是化学成分真实百分比。
#
# 四类缩写：
#   COM: calcium oxalate monohydrate，草酸钙
#   UA : uric acid，尿酸
#   MAP: magnesium ammonium phosphate，磷酸铵镁/鸟粪石
#   CYS: cystine，胱氨酸
#
# =========================================================
import pandas as pd
import numpy as np
from scipy.signal import savgol_filter

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import GroupShuffleSplit
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
