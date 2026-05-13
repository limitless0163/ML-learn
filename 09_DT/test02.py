import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import tree


# 读取数据
data = pd.read_csv('titanic.csv')
# 删除与目标变量直接相关的列，防止标签泄露
data.drop(columns=['alive'], inplace=True)

print(data.info())
print(data[:5])

feat_ranges = {}
cont_feat = ['age', 'fare']     # 连续特征
bins = 10   # 分类点数

for feat in cont_feat:
    # 数据集中存在nan,需要用np.nanmin和np.nanmax
    min_val = np.nanmin(data[feat])
    max_val = np.nanmax(data[feat])
    feat_ranges[feat] = np.linspace(min_val, max_val, bins).tolist()
    print(feat, ':')    # 查看分类点
    for spt in feat_ranges[feat]:
        print(f'{spt:.4f}')

# 只有有限取值的离散特征
cat_feat = ['pclass', 'sex', 'sibsp', 'parch', 'embarked', 'class', 'who', 'adult_male', 'deck', 'embark_town', 'alone']
for feat in cat_feat:
    data[feat] = data[feat].astype('category')  # 数据格式转为分类格式
    print(f'{feat}: {data[feat].cat.categories}')   # 查看类别
    data[feat] = data[feat].cat.codes.tolist()     # 将类别按顺序转换为整数
    ranges = list(set(data[feat]))
    ranges.sort()
    feat_ranges[feat] = ranges

# 将所有缺失值替换为-1
data.fillna(-1, inplace=True)
for feat in feat_ranges.keys():
    feat_ranges[feat] = [-1] + feat_ranges[feat]

# 划分训练集和测试集
np.random.seed(0)
feat_names = data.columns[1:]
label_name = data.columns[0]

# 重排下标后，按新的下标索引数据
data = data.reindex(np.random.permutation(data.index))
ratio = 0.8
split = int(ratio * len(data))
train_x = data[:split].drop(columns=['survived']).to_numpy()
train_y = data['survived'][:split].to_numpy()
test_x = data[split:].drop(columns=['survived']).to_numpy()
test_y = data['survived'][split:].to_numpy()
print('训练集大小：', len(train_x))
print('测试集大小：', len(test_x))
print('特征数：', train_x.shape[1])

# criterion表示分类依据，max_depth表示树的最大深度
# entropy生成的是C4.5分类树
c45 = tree.DecisionTreeClassifier(criterion='entropy', max_depth=6)
c45.fit(train_x, train_y)
# gini生成的是CART分类树
cart = tree.DecisionTreeClassifier(criterion='gini', max_depth=6)
cart.fit(train_x, train_y)

c45_train_pred = c45.predict(train_x)
c45_test_pred = c45.predict(test_x)
cart_train_pred = cart.predict(train_x)
cart_test_pred = cart.predict(test_x)
print(f'训练集准确率: C4.5: {np.mean(c45_train_pred == train_y)}, ' f'CART: {np.mean(cart_train_pred == train_y)}')
print(f'测试集准确率: C4.5: {np.mean(c45_test_pred == test_y)}, ' f'CART: {np.mean(cart_test_pred == test_y)}')