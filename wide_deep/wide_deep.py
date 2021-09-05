# author: zhangda
# date:   2021/8/29 18:45
# note:   Wide_Deep

# author:
# date: 2021/8/31 09:24
# note: wide and deep

import os
import time

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler, OrdinalEncoder
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
import tensorflow.keras.backend as K
import tensorflow as tf

data = pd.read_csv('../data/criteo_sampled_data.csv')
cols = data.columns

dense_cols = [f for f in cols if f[0] == "I"]
sparse_cols = [f for f in cols if f[0] == "C"]

# dense特征处理
def process_dense_features(data, cols):
    d = data.copy()
    for f in cols:
        d[f] = d[f].fillna(0) # d[f].fillna(d[f].mean())
        ss = StandardScaler()
        d[f] = ss.fit_transform(d[[f]]) # 输入为二维
    return d

# sparse 特征处理
def process_sparse_features(data, cols):
    d = data.copy()
    for f in cols:
        d[f] = d[f].fillna('-1') # 为啥填-1而不是0?
        label_encoder = LabelEncoder()
        d[f] = label_encoder.fit_transform(d[f])

    # 或者OrdinalEncoder()
    # for i in cols:
    #     d[f] = d[f].fillna('-1')
    # ordinal_encoder = OrdinalEncoder()
    # d[cols] = ordinal_encoder.fit_transform(d[cols])
    return d

data = process_dense_features(data, dense_cols)
data = process_sparse_features(data, sparse_cols)

## 数据集切分
# 训练集
x_train = data[:500000]
y_train = x_train.pop('label')
print("x_train.shape = ", x_train.shape)    # (500000, 40)
print("y_train.shape = ", y_train.shape)    # (500000,)

# 验证集
x_valid = data[500000:]
y_valid = x_valid.pop('label')
print("x_valid.shape = ", x_valid.shape)    # (100000, 40)
print("y_valid.shape = ", y_valid.shape)    # (100000,)


# 搭建 wide&deep模型
def wd_model(sparse_columns, dense_columns, train, test):
    ## sparse features ##
    sparse_input = []
    linear_embedding = []

    # 类别特征
    for col in sparse_columns:
        # linear_embedding
        _input = Input(shape=(1, )) # None * 1
        print('K.int_shape(_input)  ', K.int_shape(_input)) # (None, 1)
        sparse_input.append(_input)

        # 词表大小
        nums = pd.concat((train[col], test[col])).nunique() + 1
        embed = Flatten()(Embedding(nums, 4, input_length=1, embeddings_regularizer=tf.keras.regularizers.l2(0.5))(_input))  # shape=(None, 1, 4) -> shape=(None, 4)
        print('K.int_shape(embed)  ', K.int_shape(embed)) # (None, 4)

        # 放在一起，等待后续拼接
        linear_embedding.append(embed)

    # 拼接
    fst_order_sparse_layer = concatenate(linear_embedding)
    print('K.int_shape(fst_order_sparse_layer) = ', K.int_shape(fst_order_sparse_layer)) # (None, 104)

    ## dense features ##
    dense_input = []
    for col in dense_columns:
        _input = Input(shape=(1, ))
        print('K.int_shape(_input) = ', K.int_shape(_input))
        dense_input.append(_input)

    concat_dense_input = concatenate(dense_input) # (None, 1) -> (None, 13)
    print('K.int_shape(concat_dense_input) = ', K.int_shape(concat_dense_input)) # (None, 13)
    fst_order_dense_layer = Dense(4, activation='relu')(concat_dense_input)
    print('K.int_shape(fst_order_dense_layer) = ', K.int_shape(fst_order_dense_layer)) # (None, 4)

    ## linear concat ##
    linear_part = concatenate([fst_order_dense_layer, fst_order_sparse_layer]) ## (None, 4) (None, 26*4) -> (None, 4 + 26*4)
    print('K.int_shape(linear_part) = ', K.int_shape(linear_part)) # (None, 108)


    ## dnn layer ##
    fc_layer = Dropout(0.2)(Activation(activation='relu')(BatchNormalization()(Dense(128)(fst_order_dense_layer))))
    fc_layer = Dropout(0.2)(Activation(activation='relu')(BatchNormalization()(Dense(64)(fc_layer))))
    fc_layer = Dropout(0.2)(Activation(activation='relu')(BatchNormalization()(Dense(64)(fc_layer))))
    fc_layer = Dropout(0.2)(Activation(activation='relu')(BatchNormalization()(Dense(32)(fc_layer))))

    ## output layer ##
    output_layer = concatenate([linear_part, fc_layer])  # (None, 4 + 26*4 + 32)
    print('K.int_shape(output_layer) = ', K.int_shape(output_layer)) # (None, 140)

    output_layer = Dense(1, activation='sigmoid')(output_layer)
    print('K.int_shape(output_layer) = ', K.int_shape(output_layer)) # (None, 1)

    # 怎么理解？
    model = Model(inputs=sparse_input+dense_input, outputs=output_layer)

    return model


model = wd_model(sparse_cols, dense_cols, x_train, x_valid)
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=["binary_crossentropy", tf.keras.metrics.AUC(name='auc')])


# 数据格式转换 (datafram -> 二维列表)
train_sparse_x = [x_train[f].values for f in sparse_cols]
train_dense_x = [x_train[f].values for f in dense_cols]
print("train_sparse_x[0].shape = ", train_sparse_x[0].shape) # (500000,)
print("train_dense_x[0].shape = ", train_dense_x[0].shape)

train_label = [y_train.values]  # (train_len, )
valid_sparse_x = [x_valid[f].values for f in sparse_cols]
valid_dense_x = [x_valid[f].values for f in dense_cols]
valid_label = [y_valid.values]


# 训练模型
from keras.callbacks import *

# 回调函数
file_path = "wide&deep_model.h5"
earlystopping = EarlyStopping(monitor="val_loss", patience=3)
checkpoint = ModelCheckpoint(file_path, save_weights_only=True, verbose=1, save_best_only=True)

callbacks_list = [earlystopping, checkpoint]

hist = model.fit(train_sparse_x + train_dense_x,
                 train_label,
                 batch_size=512,
                 epochs=20,
                 validation_data=(valid_sparse_x + valid_dense_x, valid_label),
                 callbacks=callbacks_list,
                 shuffle=True)

np.min(hist.history['val_loss'])
np.max(hist.history['val_auc'])
