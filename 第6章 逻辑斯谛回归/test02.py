import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from sklearn.linear_model import LogisticRegression


# 从源文件中读取数据并处理
lines = np.loadtxt('lr_dataset.csv', delimiter=',', dtype=float)
x_total = lines[:, 0:2]
y_total = lines[:, 2]
print('数据集大小：', len(x_total))

# 将得到的数据在二维平面上制图，不同的类别设置不同的颜色和形状，以便于观察样本点的分布
pos_index = np.where(y_total == 1)
neg_index = np.where(y_total == 0)
plt.scatter(x_total[pos_index, 0], x_total[pos_index, 1], color='coral', marker='o', s=10)
plt.scatter(x_total[neg_index, 0], x_total[neg_index, 1], color='blue', marker='x', s=10)
plt.xlabel('x1 axis')
plt.ylabel('x2 axis')
plt.show()

# 划分训练集和测试集
np.random.seed(0)
ratio = 0.7
split = int(len(x_total) * ratio)
idx = np.random.permutation(len(x_total))
x_total = x_total[idx]
y_total = y_total[idx]
x_train = x_total[:split]
y_train = y_total[:split]
x_test = x_total[split:]
y_test = y_total[split:]

# 使用线性模型中的Logistic回归模型在数据集上训练
# 其提供的liblinear优化算法适合在较小数据集上使用
# 默认使用约束强度为1.0的L2正则化约束
lr_clf = LogisticRegression(solver='liblinear')
lr_clf.fit(x_train, y_train)
print('回归系数：', lr_clf.coef_[0], lr_clf.intercept_)

# 在数据集上用计算得到的Logistic回归模型进行预测，并计算准确率
y_pred = lr_clf.predict(x_test)
print('准确率：', np.mean(y_test == y_pred))


