import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np


# 读取原图
orig_img = np.array(mpimg.imread('origimg.jpg'), dtype=int)
orig_img[orig_img < 128] = -1   # 黑色设置为-1
orig_img[orig_img >= 128] = 1   # 白色设置为1

# 读取带噪图像
noisy_img = np.array(mpimg.imread('noisyimg.jpg'), dtype=int)
noisy_img[noisy_img < 128] = -1   # 黑色设置为-1
noisy_img[noisy_img >= 128] = 1   # 白色设置为1

def compute_noise_rate(noisy, orig):
    err = np.sum(noisy != orig)
    return err / orig.size

init_noise_rate = compute_noise_rate(noisy_img, orig_img)
print(f'带噪图像与原图不一直的像素比例: {init_noise_rate * 100:.4f}%')

# 计算坐标(i, j)处的局部能量
def compute_energy(X, Y, i, j, alpha, beta):
    # X: 当前图像
    # Y: 带噪图像
    energy = -beta * X[i][j] * Y[i][j]
    # 判断坐标是否超出边界
    if i > 0:
        energy -= alpha * X[i][j] * X[i - 1][j]
    if i < X.shape[0] - 1:
        energy -= alpha * X[i][j] * X[i + 1][j]
    if j > 0:
        energy -= alpha * X[i][j] * X[i][j - 1]
    if j < X.shape[1] - 1:
        energy -= alpha * X[i][j] * X[i][j + 1]
    return energy

# 设置超参数
alpha = 2.1
beta = 1.0
max_iter = 5

# 逐像素优化
# 复制一份噪声图像，保持网络中的Y不变，只优化X
X = np.copy(noisy_img)
for k in range(max_iter):
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            # 分别计算当前像素取1和-1时的能量
            X[i, j] = 1
            pos_energy = compute_energy(X, noisy_img, i, j, alpha, beta)
            X[i, j] = -1
            neg_energy = compute_energy(X, noisy_img, i, j, alpha, beta)
            # 将像素设置为使能量最低的值
            X[i, j] = 1 if pos_energy < neg_energy else -1

    # 展示图像并计算噪声率
    plt.figure()
    plt.axis('off')
    plt.imshow(X, cmap='binary_r')
    plt.show()

    noise_rate = compute_noise_rate(X, orig_img) * 100
    print(f'迭代次数：{k}, 噪声率：{noise_rate:.4f}%') 

