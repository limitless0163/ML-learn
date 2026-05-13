import numpy as np
import matplotlib.pyplot as plt


# 导入数据集
data = np.loadtxt('xor_dataset.csv', delimiter=',')
print('数据集大小：', len(data))
print(data[:5])

# 划分训练集和测试集
ratio = 0.8
split = int(len(data) * ratio)

np.random.seed(0)
data = np.random.permutation(data)
x_train = data[:split, :2]
y_train = data[:split, -1].reshape(-1, 1)
x_test = data[split:, :2]
y_test = data[split:, -1].reshape(-1, 1)

# 基类
class Layer:

    # 前向传播
    def forward(self, x):
        raise NotImplementedError

    # 反向传播
    def backward(self, grad):
        raise NotImplementedError
    
    # 更新函数
    def update(self, learning_rate):
        pass

class Linear(Layer):

    def __init__(self, num_in, num_out, use_bias=True):
        self.num_in = num_in        # 输入维度
        self.num_out = num_out      # 输出维度
        self.use_bias = use_bias    # 是否使用偏置项

        # 参数初始化
        # 如果把参数默认设置为0，会导致wx=0，后续计算失去意义
        # 我们直接用正态分布来初始化参数
        self.W = np.random.normal(loc=0, scale=1.0, size=(num_out, num_in))
        if use_bias:
            self.b = np.zeros((1, num_out))

    def forward(self, x):
        # 前向传播y = wx + b
        # x的维度为（batch_size, num_in）
        self.x = x
        self.y = x @ self.W.T  # y的维度为（batch_size, num_out）
        if self.use_bias:
            self.y += self.b
        return self.y
    
    def backward(self, grad):
        # 反向传播，按照链式法则计算
        # grad的维度为（batch_size, num_out）
        # 梯度要对batch_size求平均
        # grad_W的维度要与W相同，为（num_out, num_in）
        self.grad_W = grad.T @ self.x / grad.shape[0]
        if self.use_bias:
            self.grad_b = np.mean(grad, axis=0, keepdims=True)
        # 前向传播的grad的维度为（batch_size, num_in）
        grad = grad @ self.W
        return grad
    
    def update(self, learning_rate):
        # 更新参数
        self.W -= learning_rate * self.grad_W
        if self.use_bias:
            self.b -= learning_rate * self.grad_b

class Identity(Layer):
    #单位函数

    def forward(self, x):
        return x
    
    def backward(self, grad):
        return grad

class Sigmoid(Layer):
    # Logistic函数

    def forward(self, x):
        self.x = x
        self.y = 1 / (1 + np.exp(-x))
        return self.y
    
    def backward(self, grad):
        grad = grad * self.y * (1 - self.y)
        return grad

class Tanh(Layer):
    # tanh函数

    def forward(self, x):
        self.x = x
        self.y = np.tanh(x)
        return self.y
    
    def backward(self, grad):
        grad = grad * (1 - self.y ** 2)
        return grad
    
class ReLU(Layer):
    # ReLU函数

    def forward(self, x):
        self.x = x
        self.y = np.maximum(0, x)
        return self.y
    
    def backward(self, grad):
        grad = grad * (self.x >= 0)
        return grad

# 存储所有激活函数和对应名称，方便索引
activations = {
    'identity': Identity,
    'sigmoid': Sigmoid,
    'tanh': Tanh,
    'relu': ReLU
}

class MLP:
    def __init__(
        self,
        layer_sizes,
        use_bias=True,
        activation='relu',
        out_activation='identity'
    ):
        self.layers = []
        num_in = layer_sizes[0]
        for num_out in layer_sizes[1:-1]:
            # 添加全连接层
            self.layers.append(Linear(num_in, num_out, use_bias))
            # 添加激活函数
            self.layers.append(activations[activation]())
            num_in = num_out
        # 由于输入需要满足任务的一些要求，例如二分类任务需要输出值范围为[0,1]的概率值
        # 因此最后一层通常做特殊处理
        self.layers.append(Linear(num_in, layer_sizes[-1], use_bias))
        self.layers.append(activations[out_activation]())

    def forward(self, x):
        # 前向传播，依次通过每一层
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, grad):
        # 反向传播，依次通过每一层
        for layer in reversed(self.layers):
            grad = layer.backward(grad)

    def update(self, learning_rate):
        # 更新参数，依次通过每一层
        for layer in self.layers:
            layer.update(learning_rate)

# 设置超参数    
num_epochs = 1000
learning_rate = 0.1
batch_size = 128
eps=1e-7

# 创建一个层大小依次为[2, 4, 1]的MLP模型
# 对于二分类问题，我们用sigmoid函数作为输出层的激活函数，使其输出值范围为[0,1]的概率值
mlp = MLP(layer_sizes=[2, 4, 1], use_bias=True, out_activation='sigmoid')

# 训练模型
losses = []
test_losses = []
test_accs = []
for epoch in range(num_epochs):
    # 我们实现的MLP支持批量输入，因此我们采用SGD算法
    st = 0
    loss = 0.0
    while True:
        ed = min(st + batch_size, len(x_train))
        if st >= ed:
            break
        # 取出batch
        x = x_train[st:ed]
        y = y_train[st:ed]
        # 计算MLP的预测
        y_pred = mlp.forward(x)
        # 计算梯度
        grad = (y_pred - y) / (y_pred * (1 - y_pred) + eps)
        # 反向传播
        mlp.backward(grad)
        # 更新参数
        mlp.update(learning_rate)
        # 计算损失
        train_loss = np.sum(-y * np.log(y_pred + eps) - (1 - y) * np.log(1 - y_pred + eps))
        loss += train_loss
        st += batch_size

    losses.append(loss / len(x_train))
    # 计算测试集上的交叉熵和准确率
    y_pred = mlp.forward(x_test)
    test_loss = np.sum(-y_test * np.log(y_pred + eps) - (1 - y_test) * np.log(1 - y_pred + eps)) / len(x_test)
    test_acc = np.sum(np.round(y_pred) == y_test) / len(x_test)
    test_losses.append(test_loss)
    test_accs.append(test_acc)

print('测试准确率：', test_accs[-1])
# 将损失变化进行可视化
plt.figure(figsize=(16, 6))
plt.subplot(1, 2, 1)
plt.plot(losses, color='blue', label='Train Loss')
plt.plot(test_losses, color='red', ls='--', label='Test Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Cross Entropy Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(test_accs, color='red')
plt.ylim(top=1.0)
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Test Accuracy')
plt.show()

