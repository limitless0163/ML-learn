import numpy as np
import matplotlib.pyplot as plt


data = np.loadtxt('PCA_dataset.csv', delimiter=',')
print('数据集大小：', len(data))

# 可视化
plt.figure()
plt.scatter(data[:, 0], data[:, 1], color='blue', s=10)
plt.axis('square')
plt.ylim(-2, 8)
plt.grid()
plt.xlabel(r'$x_1$')
plt.ylabel(r'$x_2$')
plt.show()

def pca(X, k):
    d, m = X.shape
    if d < k:
        print('k应该要小于特征数')
        return X, None
    
    # 中心化
    X = X - np.mean(X, axis=0)
    # 计算协方差矩阵
    cov = X.T @ X
    # 计算特征值和特征向量
    eig_values, eig_vectors = np.linalg.eig(cov)
    # 获取最大的k个特征值的下标
    idx = np.argsort(-eig_values)[:k]
    # 对应的特征向量
    W = eig_vectors[:, idx]
    # 降维
    X = X @ W
    return X, W

X, W = pca(data, 2)
print('变换矩阵：\n', W)

# 绘图
plt.figure()
plt.scatter(X[:, 0], X[:, 1], color='blue', s=10)
plt.axis('square')
plt.ylim(-5, 5)
plt.grid()
plt.xlabel(r'$x_1$')
plt.ylabel(r'$x_2$')
plt.show()

