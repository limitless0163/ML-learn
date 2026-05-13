import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


data = np.loadtxt('PCA_dataset.csv', delimiter=',')
print('数据集大小：', len(data))

# 中心化
X = data - np.mean(data, axis=0)
pca_res = PCA(n_components=2).fit(X)
W = pca_res.components_.T
print('sklearn计算的变换矩阵: \n', W)
X_pca = X @ W

# 绘图
plt.figure()
plt.scatter(X_pca[:, 0], X_pca[:, 1], color='blue', s=10)
plt.axis('square')
plt.ylim(-5, 5)
plt.grid()
plt.xlabel(r'$x_1$')
plt.ylabel(r'$x_2$')
plt.show()

