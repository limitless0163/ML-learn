from matplotlib import pyplot as plt
import numpy as np
from skimage import io
from skimage.color import rgb2lab, lab2rgb
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
import os

path = 'style_transfer'

data_dir = os.path.join(path, 'vangogh')
fig = plt.figure(figsize=(16, 5))
for i, file in enumerate(np.sort(os.listdir(data_dir))[:3]):
    img = io.imread(os.path.join(data_dir, file))
    ax = fig.add_subplot(1, 3, i + 1)
    ax.imshow(img)
    ax.set_title(file)
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
plt.show()

# block_size表示向外拓展的层数，拓展1层即3*3
block_size = 1

def read_style_image(file_name, size=block_size):
    # 读入风格图像，得到映射 X->Y
    # 其中X存储3*3像素格的灰度值，Y存储中心像素格的色彩值
    # 读取图像文件，设图宽为W，高为H，得到W*H*3的RGB矩阵
    img = io.imread(file_name)
    fig = plt.figure()
    plt.imshow(img)
    plt.xlabel('X axis')
    plt.ylabel('Y axis')
    plt.show()

    # 将RGB矩阵转换为LAB矩阵,大小仍然是W*H*3，三维分别是L、A、B
    img = rgb2lab(img)
    # 取出图像的宽度和高度
    w, h = img.shape[:2]

    X = []
    Y = []
    # 枚举全部可能的中心点
    for x in range(size, w - size):
        for y in range(size, h - size):
            # 保存所有的窗口
            X.append(img[x - size:x + size + 1, y - size:y + size + 1, 0].flatten())
            # 保存窗口对应的色彩值a和b
            Y.append(img[x, y, 1:])
        return X, Y
    
X, Y = read_style_image(os.path.join(path, 'style.jpg')) # 建立映射

# weight='distance'表示邻居的权重与其到样本的距离成反比
knn = KNeighborsRegressor(n_neighbors=4, weights='distance')
knn.fit(X, Y) 

def rebuild(img, size=block_size):
    # 打印内容图像
    fig = plt.figure()
    plt.imshow(img)
    plt.xlabel('X axis')
    plt.ylabel('Y axis')
    plt.show()

    # 将内容图像转为LAB矩阵
    img = rgb2lab(img)
    w, h = img.shape[:2]

    # 初始化输出图像对应的矩阵
    photo = np.zeros((w, h, 3))
    # 枚举内容图像的中心点，保存所有窗口
    print('正在重建图像...')
    X = []
    for x in range(size, w - size):
        for y in range(size, h - size):
            # 得到中心点的对应窗口
            window = img[x - size:x + size + 1, y - size:y + size + 1, 0].flatten()
            X.append(window)
    X = np.array(X)

    # 用KNN回归器预测颜色
    print('正在预测颜色...')
    pred_ab = knn.predict(X).reshape(w - 2 * size, h - 2 * size, -1)
    # 设置输出图像
    photo[size:w - size, size:h - size, 1:] = pred_ab
    
    # 由于最外面的size层无法构建窗口，简单起见，我们直接把这些像素裁剪掉
    photo = photo[size:w - size, size:h - size, :]
    return photo

content = io.imread(os.path.join(path, 'input.jpg'))
new_photo = rebuild(content)
# 为了展示图像，我们再将其再转换为RGB矩阵
new_photo = lab2rgb(new_photo)

fig = plt.figure()
plt.imshow(new_photo)
plt.xlabel('X axis')
plt.ylabel('Y axis')
plt.show()







