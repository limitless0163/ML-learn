from tqdm import tqdm
import numpy as np
from matplotlib import pyplot as plt
from sklearn.datasets import make_classification
from sklearn.tree import DecisionTreeClassifier as DTC
from sklearn.model_selection import train_test_split


# 创建随机数据集
X, y = make_classification(
    n_samples=1000,
    n_features=16,
    n_informative=5,
    n_redundant=2,
    n_classes=2,
    flip_y=0.1,
    random_state=0
)

print(X.shape)

class RandomForest():

    def __init__(self, n_trees=10, max_features='sqrt'):
        # max_features是DTC的参数，表示节点节点分裂时随机采样的特征个数
        # sqrt代表取全部特征的平方根，None代表取全部特征，log2代表取全部特征的对数
        self.n_trees = n_trees
        self.oob_score = 0
        self.trees = [DTC(max_features=max_features) for _ in range(n_trees)]

    # 用X和y训练模型
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.n_classes = np.unique(y).shape[0]
        # 集成模型的预测，累加单个模型预测的分类概率，再取较大值作为最终分类
        ensemble = np.zeros((n_samples, self.n_classes))

        for tree in self.trees:
            # 自举采样，该采样允许重复
            idx = np.random.randint(0, n_samples, n_samples)
            # 没有被采到的样本
            unsampled_mask = np.bincount(idx, minlength=n_samples) == 0
            unsampled_idx = np.arange(n_samples)[unsampled_mask]
            # 训练当前的决策树
            tree.fit(X[idx], y[idx])
            # 累加决策树对OOB样本的预测
            ensemble[unsampled_idx] += tree.predict_proba(X[unsampled_idx])
        # 计算OOB分数，由于是分类问题，我们用准确率来衡量
        self.oob_score = np.mean(y == np.argmax(ensemble, axis=1))

    # 预测类别
    def predict(self, X):
        proba = self.predict_proba(X)
        return np.argmax(proba, axis=1)

    def predict_proba(self, X):
        # 取所有决策树预测概率的平均
        ensemble = np.mean([tree.predict_proba(X) for tree in self.trees], axis=0)
        return ensemble
    
    # 计算准确率
    def score(self, X, y):
        return np.mean(y == self.predict(X))

# 算法测试与可视化
num_trees = np.arange(1, 101, 5)
np.random.seed(0)
plt.figure()

# bagging算法
oob_score = []
train_score = []
with tqdm(num_trees) as pbar:
    for n_tree in pbar:
        rf = RandomForest(n_trees=n_tree, max_features=None)
        rf.fit(X, y)
        train_score.append(rf.score(X, y))
        oob_score.append(rf.oob_score)
        pbar.set_postfix({
            'n_tree': n_tree,
            'train_score': train_score[-1],
            'oob_score': oob_score[-1]
        })
plt.plot(num_trees, train_score, color='blue', label='bagging_train_score')
plt.plot(num_trees, oob_score, color='blue', ls='-.', label='bagging_oob_score')

# 随机森林算法
oob_score = []
train_score = []
with tqdm(num_trees) as pbar:
    for n_tree in pbar:
        rf = RandomForest(n_trees=n_tree, max_features='sqrt')
        rf.fit(X, y)
        train_score.append(rf.score(X, y))
        oob_score.append(rf.oob_score)
        pbar.set_postfix({
            'n_tree': n_tree,
            'train_score': train_score[-1],
            'oob_score': oob_score[-1]
        })
plt.plot(num_trees, train_score, color='red', ls='--', label='random_forest_train_score')
plt.plot(num_trees, oob_score, color='red', ls=':', label='random_forest_oob_score')

plt.ylabel('Score')
plt.xlabel('Number of trees')
plt.legend()
plt.show()



        
