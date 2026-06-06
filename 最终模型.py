# =========================================================
# 尿路结石 THz 谱型识别系统（1-4 THz）
# ---------------------------------------------------------
# 功能：
# 1. 读取纯品数据 + 病人数据
# 2. 自动截取 1-4 THz
# 3. SNV 标准化
# 4. SG二阶导数
# 5. PCA降维
# 6. 建立结石类型识别模型
# 7. 病人样本混合谱型概率分析
# 8. PCA可视化
#
# 作者建议：
# 不再追求精确定量
# 而是：
#   - 类型识别
#   - 混合谱型概率
# =========================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from scipy.signal import savgol_filter

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import GroupShuffleSplit

from sklearn.svm import SVC
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score
)

# =========================================================
# 参数设置
# =========================================================

PURE_FILE = 'data-纯品.csv'
PATIENT_FILE = 'data-原始.csv'

FREQ_MIN = 1.0
FREQ_MAX = 4.0

USE_SNV = True

USE_SG = True
SG_WINDOW = 11
SG_POLY = 3
SG_DERIV = 2

PCA_COMPONENTS = 8

RANDOM_STATE = 42

# =========================================================
# 读取光谱
# =========================================================

def load_spectra(filepath):

    df = pd.read_csv(filepath)

    wn = df.iloc[:, 0].values

    sample_names = df.columns[1:]

    X = df.iloc[:, 1:].values.T

    return X, sample_names, wn


# =========================================================
# 截取频段
# =========================================================

def cut_frequency(X, wn, fmin=1.0, fmax=4.0):

    mask = (wn >= fmin) & (wn <= fmax)

    X_cut = X[:, mask]

    wn_cut = wn[mask]

    return X_cut, wn_cut


# =========================================================
# SNV
# =========================================================

def snv(X):

    X_snv = []

    for x in X:

        x_new = (x - np.mean(x)) / np.std(x)

        X_snv.append(x_new)

    return np.array(X_snv)


# =========================================================
# SG二阶导数
# =========================================================

def sg_filter(X):

    return savgol_filter(
        X,
        window_length=SG_WINDOW,
        polyorder=SG_POLY,
        deriv=SG_DERIV
    )


# =========================================================
# 从样本名提取类别
# =========================================================

def get_label(name):

    name = str(name).upper()

    # =========================
    # MAP / 鸟粪石
    # =========================

    if 'AMP' in name:

        return 'MAP'

    # =========================
    # 草酸钙
    # =========================

    elif 'CAOX' in name:

        return 'COM'

    # =========================
    # 胱氨酸
    # =========================

    elif 'DCY' in name:

        return 'CYS'

    # =========================
    # 尿酸
    # =========================

    elif 'URIC' in name:

        return 'UA'

    else:

        return 'Unknown'


# =========================================================
# 预处理
# =========================================================

def preprocess(X, wn):

    # 截取频段
    X, wn = cut_frequency(
        X,
        wn,
        FREQ_MIN,
        FREQ_MAX
    )

    # SNV
    if USE_SNV:
        X = snv(X)

    # SG导数
    if USE_SG:
        X = sg_filter(X)

    return X, wn


# =========================================================
# 主程序
# =========================================================

if __name__ == '__main__':

    # =====================================================
    # 读取纯品数据
    # =====================================================

    print('=' * 60)
    print('读取纯品数据')
    print('=' * 60)

    X_pure, pure_names, wn = load_spectra(PURE_FILE)

    print('纯品数据 shape:', X_pure.shape)

    # =====================================================
    # 标签
    # =====================================================

    y = np.array([get_label(x) for x in pure_names])

    groups = []

    for name in pure_names:
        parts = name.split('-')

        # AMP-1-3
        #     ↑
        # 浓度组

        group_id = parts[1]

        groups.append(group_id)

    groups = np.array(groups)

    print('\n类别统计：')

    unique, counts = np.unique(y, return_counts=True)

    for u, c in zip(unique, counts):
        print(u, c)

    # =====================================================
    # 预处理
    # =====================================================

    X_pure, wn_cut = preprocess(X_pure, wn)

    print('\n预处理后 shape:', X_pure.shape)

    # =====================================================
    # 标准化
    # =====================================================

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(X_pure)

    # =====================================================
    # PCA
    # =====================================================

    pca = PCA(n_components=PCA_COMPONENTS)

    X_pca = pca.fit_transform(X_scaled)

    print('\nPCA完成')

    print(
        '累计解释方差:',
        np.sum(pca.explained_variance_ratio_)
    )

    # =====================================================
    # 划分训练测试
    # =====================================================

    gss = GroupShuffleSplit(
        n_splits=1,
        test_size=0.5,
        random_state=42
    )

    train_idx, test_idx = next(
        gss.split(X_pca, y, groups)
    )

    X_train = X_pca[train_idx]
    X_test = X_pca[test_idx]

    y_train = y[train_idx]
    y_test = y[test_idx]

    # =====================================================
    # SVM分类器
    # =====================================================

    clf = SVC(
        kernel='rbf',
        probability=True,
        random_state=RANDOM_STATE
    )

    clf.fit(X_train, y_train)

    # =====================================================
    # 测试集预测
    # =====================================================

    y_pred = clf.predict(X_test)

    print('\n')
    print('=' * 60)
    print('纯品分类结果')
    print('=' * 60)

    print('\nAccuracy:')

    print(
        accuracy_score(y_test, y_pred)
    )

    print('\nClassification Report:\n')

    print(
        classification_report(y_test, y_pred)
    )

    print('\nConfusion Matrix:\n')

    print(
        confusion_matrix(y_test, y_pred)
    )

    # =====================================================
    # PCA二维可视化
    # =====================================================

    pca_vis = PCA(n_components=2)

    X_vis = pca_vis.fit_transform(X_scaled)

    plt.figure(figsize=(8, 6))

    labels_unique = np.unique(y)

    for label in labels_unique:

        idx = y == label

        plt.scatter(
            X_vis[idx, 0],
            X_vis[idx, 1],
            label=label
        )

    plt.xlabel('PC1')
    plt.ylabel('PC2')

    plt.title('Stone Type PCA')

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        '纯品PCA分类.png',
        dpi=300
    )

    plt.show()

    # =====================================================
    # 病人数据分析
    # =====================================================

    print('\n')
    print('=' * 60)
    print('病人样本分析')
    print('=' * 60)

    X_patient, patient_names, wn2 = load_spectra(PATIENT_FILE)

    # 预处理
    X_patient, _ = preprocess(X_patient, wn2)

    # 使用纯品scaler
    X_patient_scaled = scaler.transform(X_patient)

    # 使用纯品PCA
    X_patient_pca = pca.transform(X_patient_scaled)

    # =====================================================
    # 概率预测
    # =====================================================

    probs = clf.predict_proba(X_patient_pca)

    class_names = clf.classes_

    # =====================================================
    # 保存结果
    # =====================================================

    result_df = pd.DataFrame()

    result_df['Sample'] = patient_names

    # 主导成分
    result_df['Predicted_Type'] = clf.predict(
        X_patient_pca
    )

    # 各类别概率
    for i, c in enumerate(class_names):

        result_df[c + '_Prob'] = probs[:, i]

    # =====================================================
    # 混合谱型判断
    # =====================================================

    mix_result = []

    for p in probs:

        sorted_idx = np.argsort(p)[::-1]

        top1 = class_names[sorted_idx[0]]
        top2 = class_names[sorted_idx[1]]

        prob1 = p[sorted_idx[0]]
        prob2 = p[sorted_idx[1]]

        # 判断是否混合
        if prob2 > 0.25:

            text = (
                f'{top1} dominant + '
                f'{top2} minor'
            )

        else:

            text = f'{top1} dominant'

        mix_result.append(text)

    result_df['Mixture_Type'] = mix_result

    # =====================================================
    # 保存
    # =====================================================

    result_df.to_csv(
        '病人结石混合谱型分析.csv',
        index=False,
        encoding='utf-8-sig'
    )

    print('\n结果预览：\n')

    print(result_df.head())

    print('\n结果已保存：')

    print('病人结石混合谱型分析.csv')

    # =====================================================
    # 病人PCA投影
    # =====================================================

    plt.figure(figsize=(9, 7))

    # 纯品
    for label in labels_unique:

        idx = y == label

        plt.scatter(
            X_vis[idx, 0],
            X_vis[idx, 1],
            alpha=0.7,
            label=f'Pure-{label}'
        )

    # 病人
    patient_vis = pca_vis.transform(X_patient_scaled)

    plt.scatter(
        patient_vis[:, 0],
        patient_vis[:, 1],
        c='black',
        marker='x',
        s=80,
        label='Patient'
    )

    plt.xlabel('PC1')
    plt.ylabel('PC2')

    plt.title('Patient Stone Projection')

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        '病人结石PCA投影.png',
        dpi=300
    )

    plt.show()

    print('\n全部完成')