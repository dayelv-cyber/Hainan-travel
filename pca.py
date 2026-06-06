import numpy as np
from sklearn.decomposition import PCA
X = np.array([
    [2.5, 2.4, 1.0, 0.5],
    [0.5, 0.7, 3.2, 2.1],
    [2.2, 2.9, 1.1, 0.4],
    [1.9, 2.2, 1.3, 0.6],
    [0.3, 0.6, 3.5, 2.4],
])
print("降维前形状",X.shape)
pca=PCA(n_components=2)
#找出主方向并投影
X_new=pca.fit_transform(X)
print("降维后形状:",X_new.shape)
print("降维后的数据:\n", X_new)
print("各主成分信息占比:", pca.explained_variance_ratio_)
print("总共保留信息:", pca.explained_variance_ratio_.sum())
