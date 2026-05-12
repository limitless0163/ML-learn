import numpy as np
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.datasets import make_friedman1
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import BaggingRegressor, RandomForestRegressor, StackingRegressor, AdaBoostRegressor


# 生成回归数据集
reg_X, reg_y = make_friedman1(
    n_samples=2000,
    n_features=100,
    noise=0.5,
    random_state=0
) 

# 划分训练集和测试集
reg_X_train, reg_X_test, reg_y_train, reg_y_test = train_test_split(reg_X, reg_y, test_size=0.2, random_state=0)

def rmse(regressor):
    # 计算regressor在测试集上的RMSE
    y_pred = regressor.predict(reg_X_test)
    return np.sqrt(np.mean((y_pred - reg_y_test) ** 2))

# XGBoost回归树
xgbr = xgb.XGBRegressor(
    n_estimators=100,
    max_depth=1,
    learning_rate=0.5,
    gamma=0.0,
    reg_lambda=0.1,
    subsample=0.5,
    objective='reg:squarederror',
    eval_metric='rmse',
    random_state=0
)

xgbr.fit(reg_X_train, reg_y_train)
print(f'XGBoost: {rmse(xgbr):.3f}')

# KNN回归
knnr = KNeighborsRegressor(n_neighbors=5).fit(reg_X_train, reg_y_train)
print(f'KNN: {rmse(knnr):.3f}')

# 线性回归
lnr = LinearRegression().fit(reg_X_train, reg_y_train)
print(f'线性回归: {rmse(lnr):.3f}')

# bagging
stump_reg = DecisionTreeRegressor(max_depth=1, min_samples_leaf=1, random_state=0)
bcr = BaggingRegressor(estimator=stump_reg, n_estimators=100, random_state=0)   # scikit-learn 1.2 把 BaggingRegressor 的 base_estimator 参数重命名为了estimator
bcr.fit(reg_X_train, reg_y_train)
print(f'bagging: {rmse(bcr):.3f}')

# 随机森林
rfr = RandomForestRegressor(n_estimators=100, max_depth=1, max_features='sqrt', random_state=0)
rfr.fit(reg_X_train, reg_y_train)
print(f'随机森林: {rmse(rfr):.3f}')

# 堆垛，默认元学习器为带L2正则化约束的线性回归
stkr = StackingRegressor(estimators=[('knn', knnr), ('ln', lnr), ('rf', rfr)])
stkr.fit(reg_X_train, reg_y_train)
print(f'Stacking: {rmse(stkr):.3f}')

# AdaBoost，回归型AdaBoost只有连续型，没有离散型
abr = AdaBoostRegressor(estimator=stump_reg, n_estimators=100, learning_rate=1.5, loss='square', random_state=0)
abr.fit(reg_X_train, reg_y_train)
print(f'AdaBoost: {rmse(abr):.3f}')

