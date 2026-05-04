import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

lines = np.loadtxt('USA_Housing.csv', delimiter=',', dtype=str)
header = lines[0]  
lines = lines[1:].astype(float)

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

# 初始化线性回归模型
linreg = LinearRegression()
# LinearRegression模型的fit方法会自动添加常数项，因此不需要手动添加一列1
linreg.fit(x_train, y_train)
# coef_是训练得到的回归系数，intercept_是常数项
print('回归系数：', linreg.coef_)
print('常数项：', linreg.intercept_)
y_pred = linreg.predict(x_test)

# 计算预测值和真实值之间的RMSE
rmse_loss = np.sqrt(np.square(y_test - y_pred).mean())
print('RMSE: ', rmse_loss)

