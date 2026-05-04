import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.neighbors import KNeighborsClassifier
from matplotlib.colors import ListedColormap

# 读入高斯数据集
data = np.loadtxt('gauss.csv', delimiter=',')
x_train = data[:, :2]
y_train = data[:, 2]
print('数据集大小：', len(x_train))

# 可视化
plt.figure()
plt.scatter(x_train[y_train == 0, 0], x_train[y_train == 0, 1], color='blue', marker='o')
plt.scatter(x_train[y_train == 1, 0], x_train[y_train == 1, 1], color='red', marker='x')
plt.xlabel('X axis')
plt.ylabel('Y axis')
plt.show()

# 设置步长
step = 0.02
# 设置网格边界
x_min, x_max = np.min(x_train[:, 0]) - 1, np.max(x_train[:, 0]) + 1
y_min, y_max = np.min(x_train[:, 1]) - 1, np.max(x_train[:, 1]) + 1
# 构造网格
xx, yy = np.meshgrid(np.arange(x_min, x_max, step), np.arange(y_min, y_max, step))
grid_data = np.concatenate((xx.reshape(-1, 1), yy.reshape(-1, 1)), axis=1)

fig = plt.figure(figsize=(16,4.5))
ks = [1, 3, 10]
cmap_light = ListedColormap(['#FFAAAA', '#AAAAFF'])

for i, k in enumerate(ks):
    # 定义KNN分类器
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(x_train, y_train)
    z = knn.predict(grid_data)

    # 画出分类结果
    ax = fig.add_subplot(1, len(ks), i+1)
    ax.pcolormesh(xx, yy, z.reshape(xx.shape), cmap=cmap_light, alpha=0.7)
    ax.scatter(x_train[y_train == 0, 0], x_train[y_train == 0, 1], color='blue', marker='o')
    ax.scatter(x_train[y_train == 1, 0], x_train[y_train == 1, 1], color='red', marker='x')

    ax.set_title(f'K = {k}')
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')

plt.show()





