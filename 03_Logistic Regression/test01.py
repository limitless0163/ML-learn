import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


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

def acc(y_true, y_pred):
    return np.mean(y_true == y_pred)

def auc(y_true, y_pred):
    # 按预测值从大到小排序，越靠前的样本预测正类概率越大
    idx = np.argsort(y_pred)[::-1]
    y_true = y_true[idx]
    y_pred = y_pred[idx]
    # 把y_pred中不重复的值当作阈值，依次计算FP样本和TP样本的数量
    # 由于两个数组已经排序且位置对应，直接从前向后累加即可
    tp = np.cumsum(y_true)
    fp = np.cumsum(1 - y_true)
    tpr = tp / tp[-1]
    fpr = fp / fp[-1]
    # 依次枚举FPR，计算曲线下面积
    # 方便起见，给曲线加上(0, 0)
    s = 0.0
    tpr = np.concatenate(([0], tpr))
    fpr = np.concatenate(([0], fpr))
    for i in range(1, len(tpr)):
        s += (fpr[i] - fpr[i - 1]) * (tpr[i])
    return s

# Logistic函数
def logistic(z):
    return 1 / (1 + np.exp(-z))

def GD(num_steps, learning_rate, l2_coef):
    # 初始化参数模型
    theta = np.random.normal(size=(X.shape[1],))
    train_losses = []
    test_losses = []
    train_acc = []
    test_acc = []
    train_auc = []
    test_auc = []
    for i in range(num_steps):
        pred = logistic(X @ theta)
        grad = -X.T @ (y_train - pred) + l2_coef * theta
        theta -= learning_rate * grad
        # 记录损失函数
        train_loss = - y_train.T @ np.log(pred) - (1 - y_train).T @ np.log(1 - pred) + l2_coef * np.linalg.norm(theta) ** 2 / 2
        train_losses.append(train_loss / len(X))
        test_pred = logistic(X_test @ theta)
        test_loss = - y_test.T @ np.log(test_pred) - (1 - y_test).T @ np.log(1 - test_pred)
        test_losses.append(test_loss / len(X_test))
        # 记录各个评价指标，阈值采用0.5
        train_acc.append(acc(y_train, pred >= 0.5))
        test_acc.append(acc(y_test, test_pred >= 0.5))
        train_auc.append(auc(y_train, pred))
        test_auc.append(auc(y_test, test_pred))

    return theta, train_losses, test_losses, train_acc, test_acc, train_auc, test_auc

# 定义梯度下降迭代的次数，学习率，以及L2正则化约束强度
num_steps = 250
learning_rate = 0.002
l2_coef = 1.0
np.random.seed(0)

# 在x矩阵上拼接1
X = np.concatenate((x_train, np.ones((x_train.shape[0], 1))), axis=1)
X_test = np.concatenate((x_test, np.ones((x_test.shape[0], 1))), axis=1)

theta, train_losses, test_losses, train_acc, test_acc, train_auc, test_auc = GD(num_steps, learning_rate, l2_coef)

# 计算测试集上的预测准确率
y_pred = np.where(logistic(X_test @ theta) >= 0.5, 1, 0)
final_acc = acc(y_test, y_pred)
print('预测准确率：', final_acc)
print('回归系数：', theta)

plt.figure(figsize=(13, 9))
xticks = np.arange(num_steps) + 1
# 绘制训练曲线
plt.subplot(2, 2, 1)
plt.plot(xticks, train_losses, color='blue', label='train loss')
plt.plot(xticks, test_losses, color='red', ls='--', label='test loss')
plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

# 绘制准确率曲线
plt.subplot(2, 2, 2)
plt.plot(xticks, train_acc, color='blue', label='train accuracy')
plt.plot(xticks, test_acc, color='red', ls='--', label='test accuracy')
plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

# 绘制AUC曲线
plt.subplot(2, 2, 3)
plt.plot(xticks, train_auc, color='blue', label='train AUC')
plt.plot(xticks, test_auc, color='red', ls='--', label='test AUC')
plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
plt.xlabel('Epochs')
plt.ylabel('AUC')
plt.legend()

# 绘制决策边界
plt.subplot(2, 2, 4)
plot_x = np.linspace(-1.1, 1.1, 100)
# 直线方程：theta[0] * x1 + theta[1] * x2 + theta[2] = 0
plot_y = -(theta[0] * plot_x + theta[2]) / theta[1]
pos_index = np.where(y_total == 1)
neg_index = np.where(y_total == 0)
plt.scatter(x_total[pos_index, 0], x_total[pos_index, 1], color='coral', marker='o', s=10)
plt.scatter(x_total[neg_index, 0], x_total[neg_index, 1], color='blue', marker='x', s=10)
plt.plot(plot_x, plot_y, color='green', ls='-.')
plt.xlim(-1.1, 1.1)
plt.ylim(-1.1, 1.1)
plt.xlabel('x1 axis')
plt.ylabel('x2 axis')
plt.show()




    