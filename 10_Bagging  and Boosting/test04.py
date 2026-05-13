from tqdm import tqdm
import numpy as np
from matplotlib import pyplot as plt
from sklearn.datasets import make_classification
from sklearn.tree import DecisionTreeClassifier as DTC
from sklearn.model_selection import train_test_split
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression as LR
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.neighbors import KNeighborsClassifier as KNC
from sklearn.ensemble import AdaBoostClassifier


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

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# 初始化stump
stump = DTC(max_depth=1, min_samples_leaf=1, random_state=0)

# 弱分类器个数
M = np.arange(1, 101, 5)
bg_score = []
rf_score = []
dsc_ada_score = []
real_ada_score = []
plt.figure()

with tqdm(M) as pbar:
    for m in pbar:
        # bagging算法
        bc = BaggingClassifier(estimator=stump, n_estimators=m, random_state=0)
        bc.fit(X_train, y_train)
        bg_score.append(bc.score(X_test, y_test))
        # 随机森林算法
        rfc = RandomForestClassifier(n_estimators=m, max_depth=1, min_samples_leaf=1, random_state=0)
        rfc.fit(X_train, y_train)
        rf_score.append(rfc.score(X_test, y_test))
        # 离散Adaboost,SAMME时分布加性模型的缩写
        dsc_adaboost = AdaBoostClassifier(estimator=stump, n_estimators=m, algorithm='SAMME', random_state=0)
        dsc_adaboost.fit(X_train, y_train)
        dsc_ada_score.append(dsc_adaboost.score(X_test, y_test))
        # 实Adaboost，SAMME.R表示弱分类器输出实数，新版已经删除
        real_adaboost = AdaBoostClassifier(estimator=stump, n_estimators=m, algorithm='SAMME', random_state=0)
        real_adaboost.fit(X_train, y_train)
        real_ada_score.append(real_adaboost.score(X_test, y_test))

# 绘图
plt.plot(M, bg_score, color='blue', label='Bagging')
plt.plot(M, rf_score, color='red', ls='--', label='Random Forest')
plt.plot(M, dsc_ada_score, color='green', ls='-.', label='Discrete Adaboost')
plt.plot(M, real_ada_score, color='purple', ls=':', label='Real Adaboost')
plt.xlabel('Number of trees')
plt.ylabel('Test score')
plt.legend()
plt.show()

