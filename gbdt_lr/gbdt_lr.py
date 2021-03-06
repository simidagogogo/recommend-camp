# author: zhangda
# date:   2021/8/29 12:03
# note:   GBDT+LR

import pandas as pd
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder

data = pd.read_csv('criteo_sampled_data.csv')[0:1000]
cols = data.columns
print(cols)
print(data.head())

dense_cols = [col for col in cols if col[0] == 'I']
sparse_cols = [col for col in cols if col[0] == 'C']
print(dense_cols)
print(sparse_cols)

# 类别特征的两两特征交叉n(n-1)/2
def process_sparse_features(data, cols):
    # 先深拷贝一份数据出来，防止直接修改入参
    d = data.copy()
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            f1 = cols[i]
            f2 = cols[j]
            f3 = f1 + f2

            d[f1] = d[f1].fillna('-1')
            d[f2] = d[f2].fillna('-1')
            d[f3] = d[f1].astype(str).values + '_' + d[f2].astype(str).values

            label_encoder = LabelEncoder()
            d[f1] = label_encoder.fit_transform(d[f1])
            d[f3] = label_encoder.fit_transform(d[f3])
    return d

data = process_sparse_features(data, sparse_cols)
print(data.shape)

# 1.为什么要删除C26这个问题，process_sparse_feats内部其实是在做二阶交叉的，i = 25的时候，第二层for循环是for i in range(26,26)，不执行，所以C26没做label_encoder
# 删除掉colomns='C26'的列数据
del data['C26']

# 训练集
x_train = data[:500]
y_train = x_train.pop('label')
print("x_train = ", x_train)

# 验证集
x_valid = data[500:]
y_valid = x_valid.pop('label')

## LightGBM
n_estimators = 50   # 50 棵树
num_leaves = 64     # 每棵树64个叶子节点

# 开始训练gbdt (可能写的有问题)
# model = lgb.LGBMRegressor(objective='binary',
#                           n_estimators=50,
#                           num_leaves=64,
#                           learning_rate=0.1,
#                           subsample=0.8,
#                           colsample_bytree=0.8,
#                           min_child_weight=0.5,
#                           random_state=2020)
# model.fit(x_train, y_train,
#           eval_set=[(x_train, y_train), (x_valid, y_valid)],
#           eval_names=['train', 'valid'],
#           eval_metric='binary_logloss',
#           verbose=10)

# 开始训练gbdt
model = lgb.LGBMClassifier(objective='binary',
                          subsample= 0.8,
                          colsample_bytree= 0.8,
                          min_child_weight= 0.5,
                          num_leaves=num_leaves,
                          learning_rate=0.1,
                          n_estimators=n_estimators,
                          random_state = 2020)

# 注意：训练仅仅使用(x_train, y_train), 所以前面需要切分数据集
model.fit(x_train, y_train,
          eval_set = [(x_train, y_train), (x_valid, y_valid)],
          eval_names = ['train', 'valid'],
          eval_metric = 'binary_logloss',
          # categorical_feature = sparse_cols,
          early_stopping_rounds=100,
          verbose=10)

# 提取叶子节点
# 得到每一条训练数据落在了每棵树的哪个叶子结点上（pred_leaf = True 表示返回每棵树的叶节点序号）
gbdt_features_train = model.predict(x_train, pred_leaf=True) # pred_leaf : bool, optional (default=False) Whether to predict leaf index.
print("gbdt_features_train = ", gbdt_features_train)

# 打印结果的 shape
print(gbdt_features_train.shape) # (500000, 50)，50表示50颗树，值为样本所落入的某棵树的某个叶子节点索引

# 打印前5个数据
print(gbdt_features_train[:5])

# 同样要获取测试集的叶节点索引
gbdt_features_valid = model.predict(x_valid, pred_leaf=True)


# 将 50 颗树的叶节点序号构造成 DataFrame，方便后续进行 one-hot
gbdt_features_name = ['gbdt_leaf_' + str(i) for i in range(n_estimators)]
print(gbdt_features_name)

df_train_gbdt_features = pd.DataFrame(gbdt_features_train, columns=gbdt_features_name)
df_valid_gbdt_features = pd.DataFrame(gbdt_features_valid, columns=gbdt_features_name)

train_len = df_train_gbdt_features.shape[0]
print("train_len = ", train_len)

data = pd.concat([df_train_gbdt_features, df_valid_gbdt_features])

# 对每棵树的叶节点序号进行 one-hot
for col in gbdt_features_name:
    onehot_feats = pd.get_dummies(data[col], prefix = col)
    print("onehot_feats.shape = ", onehot_feats.shape)

    # 删除原始的特征列
    data.drop([col], axis = 1, inplace = True)

    # 加入one-hot特征列
    data = pd.concat([data, onehot_feats], axis = 1)

# 切分数据集
train = data[: train_len]
valid = data[train_len:]


### 开始训练lr ####
lr = LogisticRegression(C=5, solver='sag')
lr.fit(train, y_train)

# 评价指标
train_logloss = log_loss(y_train, lr.predict_proba(train)[:, 1])
print('tr-logloss: ', train_logloss)

valid_logloss = log_loss(y_valid, lr.predict_proba(valid)[:, 1])
print('val-logloss: ', valid_logloss)

auc = roc_auc_score(y_valid, lr.predict_proba(valid)[:, 1])
print("auc = ", auc)
