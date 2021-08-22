# author: zhangda
# date: 2021-08-22
# note: gbdt + lr

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import lightgbm as lgb 
from sklearn.metrics import log_loss

import warnings
warnings.filterwarnings('ignore')

import gc

# GBDT部分

# 整合数据集
def concordanceDataset(path):
    print('read data: 读取train和test数据集')
    df_train = pd.read_csv(path + 'train.csv')
    df_test = pd.read_csv(path + 'test.csv')
    print('read data end')

    # 丢掉Id列
    df_train.drop(['Id'], axis = 1, inplace = True)
    df_test.drop(['Id'], axis = 1, inplace = True)

    # 给测试集打上标签-1
    df_test['Label'] = -1

    # 将训练集和测试集concat在一起, 统一进行OneHot等处理后再拆分
    # 这里没有指定axis = 1, 应该是axis = 0. 即按行拼接
    data = pd.concat([df_train, df_test])

    # 所有空值使用-1填充
    data = data.fillna(-1)

    return data


# 类别特征OneHot
def featureOneHot4GBDT(data):
    """
    类别特征处理
    :param data: 原始数据
    :return: OneHot之后的数据，12042维特征 + 1维标签
    """

    # data.to_csv('data/data.csv', index = False)
    #
    # 连续特征
    continuous_feature = ['I'] * 13
    continuous_feature = [col + str(i + 1) for i, col in enumerate(continuous_feature)]
    print('continuous_feature',continuous_feature)


    # 类别特征
    # 1. 26维类别特征对应列名
    category_feature = ['C'] * 26
    category_feature = [col + str(i + 1) for i, col in enumerate(category_feature)]
    print('category_feature', category_feature)

    # discrite one-hot encoding
    # 2. 类别特征OneHot
    print('begin one-hot:')
    for col in category_feature:
        # 生成独热编码
        onehot_features = pd.get_dummies(data[col], prefix = col)

        # 原始特征列删掉
        data.drop([col], axis = 1, inplace = True)

        # axis = 1表示列拼接
        data = pd.concat([data, onehot_features], axis = 1)
    print('one-hot end')

    # data.to_csv('./data/dataProcessed.csv', index=False)
    return data


# 样本构建
def buildSamples(data):
    """
    拆分训练集和测试集，训练集部分进一步拆分为训练集和验证集
    :param data: 包含训练集和测试集的样本
    :return: 训练集、验证集、测试集、训练集标签
    """

    train = data[data['Label'] != -1]
    target = train.pop('Label')

    test = data[data['Label'] == -1]
    test.drop(['Label'], axis = 1, inplace = True)

    print('split train and testset:')
    x_train, x_valid, y_train, y_valid = train_test_split(train, target, test_size = 0.2, random_state = 2018)

    return x_train, x_valid, y_train, y_valid, train, test, target


# 模型训练与验证
def modelTrainAndValidOfGDBT(x_train, x_valid, y_train, y_valid):
    print('begin train gbdt:')
    gbm = lgb.LGBMRegressor(objective='binary',
                            subsample= 0.8,
                            min_child_weight= 0.5,
                            colsample_bytree= 0.7,
                            num_leaves=100,
                            max_depth = 12,
                            learning_rate=0.05,
                            n_estimators=10)

    gbm.fit(x_train, y_train,
            eval_set = [(x_train, y_train), (x_valid, y_valid)],
            eval_names = ['train', 'valid'],
            eval_metric = 'binary_logloss'
            # early_stopping_rounds = 100,
            )
    model = gbm.booster_

    return model


# 预测并获取叶子节点
def predictAndGetLeaf(model, train, test):
    """
    验证集和测试集上分别预测、获取叶子节点编号
    :param model:   训练好的gdbt模型
    :param train:   训练集（未拆分训练集和验证集）
    :param test:    测试集
    :return:    训练集数量(用户后续按原切分方式切分)、用于LR模型训练的新数据集、GBDT投喂给LR部分的特征列名
    """

    print('train to get leaf: 获取的是每个样本在每棵树上最终落到了第几个叶子节点上')
    gbdt_features_train = model.predict(train, pred_leaf = True)
    gbdt_features_test = model.predict(test, pred_leaf = True)

    # 这里怎么理解？
    print("gbdt_features_train.shape[0]", gbdt_features_train.shape[0])
    print("gbdt_features_train.shape[1]", gbdt_features_train.shape[1])

    gbdt_features_name = ['gbdt_leaf_' + str(i) for i in range(gbdt_features_train.shape[1])]
    print("gbdt_features_name", gbdt_features_name)

    df_train_gbdt_features = pd.DataFrame(gbdt_features_train, columns = gbdt_features_name)
    print('df_train_gbdt_features', df_train_gbdt_features)

    df_test_gbdt_features = pd.DataFrame(gbdt_features_test, columns = gbdt_features_name)
    print('df_test_gbdt_features', df_test_gbdt_features)

    # 构建用户训练LR模型的数据集，同样先合并再拆分
    print('create new dataset: 用于LR模型训练')
    train = pd.concat([train, df_train_gbdt_features], axis = 1)
    test = pd.concat([test, df_test_gbdt_features], axis = 1)

    train_len = train.shape[0]
    print("train_len", train_len)

    data = pd.concat([train, test])

    # 释放内存
    del train
    del test
    gc.collect()

    return train_len, data, gbdt_features_name


# LR部分
# 叶子节点index部分OneHot
def featureOneHot4LR(gbdt_features_name, data, train_len):
    """
    此时的样本包含12536维特征和1维标签
    :return:
    """

    # leafs one-hot
    print('begin one-hot:')
    for col in gbdt_features_name:
        print('this is feature:', col)

        # 生成one-hot特征
        onehot_features = pd.get_dummies(data[col], prefix=col)

        # 丢掉原始的特征
        data.drop([col], axis=1, inplace=True)

        # 原始特征拼接one-hot特征
        data = pd.concat([data, onehot_features], axis=1)
    print('one-hot ending')

    # 划分训练集和测试集
    train = data[:train_len]
    test = data[train_len:]

    del data
    gc.collect()

    return train, test


# 模型训练与验证
def modelTrainAndValidOfLR(train, target, test):
    """
    LR模型训练、验证与预测
    :param train:   lr模型的训练集特征部分
    :param target:  lr模型训练集标签
    :param test:    lr模型测试集，样本与给定的保持一致
    :return:
    """

    x_train, x_valid, y_train, y_valid = train_test_split(train, target, test_size = 0.3, random_state = 2018)

    # lr
    print('beging train lr:')
    lr = LogisticRegression()
    lr.fit(x_train, y_train)
    train_logloss = log_loss(y_train, lr.predict_proba(x_train)[:, 1])
    print('train-logloss: ', train_logloss)

    valid_logloss = log_loss(y_valid, lr.predict_proba(x_valid)[:, 1])
    print('valid-logloss: ', valid_logloss)

    print('begin predict:')
    y_pred = lr.predict_proba(test)[:, 1]

    print('write log:')
    res = pd.read_csv('data/test.csv')
    log = pd.DataFrame({'Id': res['Id'], 'Label': y_pred})
    log.to_csv('log/log_gbdt+lr_trlogloss_%s_vallogloss_%s.csv' % (train_logloss, valid_logloss), index = False)
    print('end')

if __name__ == '__main__':

    # 1.整合数据集用于GBDT模型训练
    dataOriginal4gbdt = concordanceDataset('./data/')
    dataProcessed4gbdt = featureOneHot4gdbt(dataOriginal4gbdt)

    # 2.用于GBDT模型训练、验证、预测的样本构建
    x_train, x_val, y_train, y_val, train, test, target = buildSamples(dataProcessed4gbdt)

    # 3.GBDT部分模型训练
    model = modelTrainAndValidOfGDBT(x_train, y_train, x_val, y_val)

    # 4.GBDT与LR衔接部分
    train_len, samples4lr, gbdt_feats_name = predictAndGetLeaf(model, train, test)

    # 5.LR模型部分类别特征OneHot
    train4lr, test4lr = featureOneHot4LR(gbdt_feats_name, samples4lr, train_len)

    # 6.LR模型训练与验证
    modelTrainAndValidOfLR(train4lr, target, test4lr)

    """
    read data:
    read data end
    
    category_feature ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10', 'C11', 'C12', 'C13', 'C14', 'C15', 'C16', 'C17', 'C18', 'C19', 'C20', 'C21', 'C22', 'C23', 'C24', 'C25', 'C26']
    
    begin one-hot:
    one-hot end
    
    split train and testset:
    
    begin train gbdt:
    [1]	train's binary_logloss: 0.487698	val's binary_logloss: 0.562164
    [2]	train's binary_logloss: 0.476977	val's binary_logloss: 0.559033
    [3]	train's binary_logloss: 0.466493	val's binary_logloss: 0.554616
    [4]	train's binary_logloss: 0.456887	val's binary_logloss: 0.553949
    [5]	train's binary_logloss: 0.448791	val's binary_logloss: 0.550368
    [6]	train's binary_logloss: 0.440923	val's binary_logloss: 0.550688
    [7]	train's binary_logloss: 0.433358	val's binary_logloss: 0.54867
    [8]	train's binary_logloss: 0.42576	    val's binary_logloss: 0.546644
    [9]	train's binary_logloss: 0.418043	val's binary_logloss: 0.544487
    [10]train's binary_logloss: 0.410371	val's binary_logloss: 0.543174
    
    train to get leaf: 获取的是每个样本在每棵树上最终落到了第几个叶子节点上
    
    create new dataset: 用于LR模型训练
    begin one-hot:
    this is feature: gbdt_leaf_0
    this is feature: gbdt_leaf_1
    this is feature: gbdt_leaf_2
    this is feature: gbdt_leaf_3
    this is feature: gbdt_leaf_4
    this is feature: gbdt_leaf_5
    this is feature: gbdt_leaf_6
    this is feature: gbdt_leaf_7
    this is feature: gbdt_leaf_8
    this is feature: gbdt_leaf_9
    one-hot ending
    
    beging train lr:
    train-logloss:  0.1744764219620596
    valid-logloss:  0.5094803030311569
    begin predict on test:
    write log:
    end
    """
