from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator
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

# 该函数每次返回大小为batch_size的批量
# x和y分别是输入和标签
# 若shuffle = True，则每次遍历时会将数据重新随机划分
def batch_generator(x, y, batch_size, shuffle=True):
    # 批量计数器
    batch_count = 0
    if shuffle:
        # 随机生成0到len(x)-1的下标
        idx = np.random.permutation(len(x))
        x = x[idx]
        y = y[idx]
    while True:
        start = batch_count * batch_size
        end = min(start + batch_size, len(x))
        if start >= end:
            # 已经遍历一遍，结束生成
            break
        batch_count += 1
        yield x[start:end], y[start:end]

def SGD(num_epochs, learning_rate, batch_size):
    # 拼接原始矩阵
    X = np.concatenate((x_train, np.ones((len(x_train), 1))), axis=-1) 
    X_test = np.concatenate((x_test, np.ones((len(x_test), 1))), axis=-1)
    # 随机初始化参数
    theta = np.random.normal(size=X.shape[1])

    # 随机梯度下降
    # 为了观察迭代过程，我们记录每一次迭代后在训练集和测试集上的RMSE
    train_losses = []
    test_losses = []
    for i in range(num_epochs):
        # 初始化批量生成器
        batch_g = batch_generator(X, y_train, batch_size, shuffle=True)
        train_loss = 0
        for x_batch, y_batch in batch_g:
            # 计算梯度
            grad = x_batch.T @ (x_batch @ theta - y_batch)
            # 更新参数
            theta -= learning_rate * grad / len(x_batch)
            # 累加平方误差
            train_loss += np.square(x_batch @ theta - y_batch).sum()
        # 计算训练和测试误差
        train_loss = np.sqrt(train_loss / len(X))
        train_losses.append(train_loss)
        test_loss = np.sqrt(np.square(X_test @ theta - y_test).mean())
        test_losses.append(test_loss)

    # 输出结果，绘制训练曲线
    print('回归系数：', theta)
    return theta, train_losses, test_losses

# 设置轮数、学习率与批量大小
num_epochs = 20
learning_rate = 0.01
batch_size = 32
# 设置随机种子
np.random.seed(0)

_, train_losses, test_losses = SGD(num_epochs, learning_rate, batch_size)

# 将损失函数关于轮数的关系制图，可以看到损失函数先一直保持下降，之后趋于平稳
plt.plot(np.arange(num_epochs), train_losses, color='blue', label='Train Loss')
plt.plot(np.arange(num_epochs), test_losses, color='orange', ls='--',  label='Test Loss')
# 由于epoch是整数，这里把图中的横坐标也设置为整数
plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
plt.xlabel('Epoch')
plt.ylabel('RMSE')
plt.legend()
plt.show()

_, loss1, _ = SGD(num_epochs, learning_rate=0.1, batch_size=batch_size)
_, loss2, _ = SGD(num_epochs, learning_rate=0.001, batch_size=batch_size)
plt.plot(np.arange(num_epochs), loss1, color='blue', label='lr = 0.1')
plt.plot(np.arange(num_epochs), train_losses, color='green', ls='--', label='lr = 0.01')
plt.plot(np.arange(num_epochs), loss2, color='orange', ls='-.', label='lr = 0.001')
plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
plt.xlabel('Epoch')
plt.ylabel('RMSE')
plt.legend()
plt.show()

_, loss3, _ = SGD(num_epochs, learning_rate=1.5, batch_size=batch_size)
print('最终损失：', loss3[-1])
plt.plot(np.arange(num_epochs), np.log(loss3), color='blue', label='lr = 1.5')
plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
plt.xlabel('Epoch')
plt.ylabel('Log RMSE')
plt.legend()
plt.show()