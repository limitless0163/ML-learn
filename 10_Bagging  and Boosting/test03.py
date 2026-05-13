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

# 堆垛分类器，继承sklearn中的集成分类器基类EnsembleClassifier
class StackingClassifier():

    def __init__(
        self,
        classifiers,    # 基分类器
        meta_classifier,    # 元分类器
        concat_feature=False,   # 是否将原始样本拼接在新数据上
        kfold=5     # K折交叉验证
    ):
        self.classifiers = classifiers
        self.meta_classifier = meta_classifier
        self.concat_feature = concat_feature
        self.kf = KFold(n_splits=kfold)
        # 为了在测试时计算平均，我们需要保留每个分类器
        self.k_fold_classifiers = []

    def fit(self, X, y):
        # 用X和y训练基分类器和元分类器
        n_samples, n_features = X.shape
        self.n_classes = np.unique(y).shape[0]

        if self.concat_feature:
            features = X
        else:
            features = np.zeros((n_samples, 0))
        for classifier in self.classifiers:
            self.k_fold_classifiers.append([])
            # 训练每个基分类器
            predict_proba = np.zeros((n_samples, self.n_classes))
            for train_idx, test_idx in self.kf.split(X):
                # 交叉验证
                clf = clone(classifier)
                clf.fit(X[train_idx], y[train_idx])
                predict_proba[test_idx] = clf.predict_proba(X[test_idx])
                self.k_fold_classifiers[-1].append(clf)
            features = np.concatenate([features, predict_proba], axis=-1)
        # 训练元分类器
        self.meta_classifier.fit(features, y)

    def _get_features(self, X):
        # 计算输入X的特征
        if self.concat_feature:
            features = X
        else:
            features = np.zeros((X.shape[0], 0))
        for k_classifiers in self.k_fold_classifiers:
            k_feat = np.mean([clf.predict_proba(X) for clf in k_classifiers], axis=0)
            features = np.concatenate([features, k_feat], axis=-1)
        return features

    def predict(self, X):
        return self.meta_classifier.predict(self._get_features(X))

    def score(self, X, y):
        return self.meta_classifier.score(self._get_features(X), y)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# 基分类器
rf = RFC(n_estimators=10, max_features='sqrt', random_state=0).fit(X_train, y_train)
knc = KNC().fit(X_train, y_train)
# multi_class='ovr'表示二分类问题
lr = LR(solver='liblinear', multi_class='ovr', random_state=0).fit(X_train, y_train)
print('随机森林:', rf.score(X_test, y_test))
print('KNN:', knc.score(X_test, y_test))
print('Logistic:', lr.score(X_test, y_test))
# 元分类器
meta_lr = LR(solver='liblinear', multi_class='ovr', random_state=0)

sc = StackingClassifier([rf, knc, lr], meta_lr, concat_feature=False)
sc.fit(X_train, y_train)
print('Stacking分类器:', sc.score(X_test, y_test))

# 带原始特征的stacking分类器
sc_concat = StackingClassifier([rf, knc, lr], meta_lr, concat_feature=True)
sc_concat.fit(X_train, y_train)
print('带原始特征的Stacking分类器:', sc_concat.score(X_test, y_test))


