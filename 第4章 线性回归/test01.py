import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from sklearn.preprocessing import StandardScaler

# 从源文件加载数据，并输出和查看数据的各项特征
lines = np.loadtxt('USA_Housing.csv', delimiter=',', dtype=str)
header = lines[0]  
lines = lines[1:].astype(float)
print('数据特征：', ', '.join(header[:-1]))
print('数据标签：', header[-1])
print('数据总条数：', len(lines))

# 划分训练集和测试集
ratio = 0.8
split = int(len(lines) * ratio)
lines = np.random.permutation(lines)  # 打乱数据顺序
train = lines[:split]
test = lines[split:]

# 数据标准化
scaler = StandardScaler()
scaler.fit(train) # 只使用训练集数据来计算均值和方差
train = scaler.transform(train)
test = scaler.transform(test)

# 划分输入和标签
x_train = train[:, :-1]
y_train = train[:, -1].flatten()
x_test = test[:, :-1]
y_test = test[:, -1].flatten()

# 在X矩阵最后添加一列1，代表常数项
X = np.concatenate((x_train, np.ones((len(x_train), 1))), axis=-1)
# @ 表示矩阵相乘，X.T表示矩阵X的转置，np.linalg.inv表示矩阵求逆
theta = np.linalg.inv(X.T @ X) @ X.T @ y_train
print('回归系数：', theta)

# 在测试集上使用回归系数进行预测
X_test = np.concatenate((x_test, np.ones((len(x_test), 1))), axis=-1)
y_pred = X_test @ theta

# 计算预测值和真实值之间的RMSE
rmse_loss = np.sqrt(np.square(y_test - y_pred).mean())
print('RMSE: ', rmse_loss)

