from tqdm import tqdm
import numpy as np
from matplotlib import pyplot as plt
from sklearn.datasets import make_classification
from sklearn.tree import DecisionTreeClassifier as DTC
from sklearn.model_selection import train_test_split
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier


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

bc = BaggingClassifier(n_estimators=100, oob_score=True, random_state=0)
bc.fit(X, y)
print('bagging:', bc.oob_score_)

rfc = RandomForestClassifier(n_estimators=100, max_features='sqrt', oob_score=True, random_state=0)
rfc.fit(X, y)
print('随机森林：', rfc.oob_score_)

