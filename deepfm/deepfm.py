# author: zhangda
# date:   2021/9/2 07:23
# note:   2021-08-28 ctr代码 王老师
#

import os
import time
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
from tensorflow.keras.layers import *
import tensorflow.keras.backend as K
import tensorflow as tf
from tensorflow.keras.models import Model

data = pd.read_csv('../data/criteo_sampled_data.csv')
cols = data.columns
print('cols = ', cols)

dense_cols = [f for f in cols if f[0] == 'I']
sparse_cols = [f for f in cols if f[0] == 'C']


# dense特征处理
def process_dense_features(data, cols):
    d = data.copy()
    for f in cols:
        d[f] = d[f].fillna(0)
        ss = StandardScaler()
        d[f] = ss.fit_transform(d[[f]])  # 2d
    return d


# sparse特征处理
def process_sparse_features(data, cols):
    d = data.copy()
    for f in cols:
        d[f] = d[f].fillna('-1')
        label_encoder = LabelEncoder()
        d[f] = label_encoder.fit_transform(d[f])  # 1d
    return d


data = process_dense_features(data, dense_cols)
print(data.head())
data = process_sparse_features(data, sparse_cols)
print(data.head())

# 数据集切分
x_train = data[:500000]
y_train = x_train.pop('label')

x_valid = data[500000:]
y_valid = x_valid.pop('label')


# 搭建 deepfm 模型
def deepfm_model(sparse_columns, dense_columns, train, test):
    ####### sparse features ##########

    # 存放输入tensor的列表
    sparse_input = []

    # 存放一阶embedding特征的列表
    linear_embedding = []

    # 存放二阶fm隐向量的列表
    fm_embedding = []

    # 类别特征
    for col in sparse_columns:
        ## linear_embedding
        _input = Input(shape=(1,))  # 表示一维向量，且仅有一个元素
        print("K.int_shape(_input) = ", K.int_shape(_input))  # (None, 1)
        sparse_input.append(_input)

        # 特征空间维度
        nums = pd.concat([train[col], test[col]], axis=0).nunique() + 1
        embed = Flatten()(Embedding(nums, 1, embeddings_regularizer=tf.keras.regularizers.l2(0.5))(_input))
        print('K.int_shape(embed) = ', K.int_shape(embed))
        linear_embedding.append(embed)

        ## fm_embedding FM的隐向量
        embed = Embedding(nums, 10, input_length=1, embeddings_regularizer=tf.keras.regularizers.l2(0.5))(_input)
        reshape = Reshape(target_shape=(10,))(embed)  # None * 1 * 10 -> None * 10 ?
        fm_embedding.append(reshape)

    ## fst_order_sparse_layer ##
    fst_order_sparse_layer = concatenate(linear_embedding)

    ####### fm layer ########## （这里只有FM隐向量之间交叉么？好像没有和dnn部分的隐向量交叉？？）
    fm_square = Lambda(lambda x: K.square(x))(Add()(fm_embedding))  #
    square_fm = Add()([Lambda(lambda x: K.square(x))(embed) for embed in fm_embedding])

    # FM的隐向量经过FM内积得到的交叉特征
    snd_order_sparse_layer = subtract([fm_square, square_fm])
    snd_order_sparse_layer = Lambda(lambda x: x * 0.5)(snd_order_sparse_layer)

    ####### dense features ##########
    # 稠密特征
    dense_input = []
    for col in dense_columns:
        _input = Input(shape=(1,))
        dense_input.append(_input)

    # 为啥13个稠密特征要经过MLP层压缩为4维？？
    concat_dense_input = concatenate(dense_input, axis=-1)

    ## fst_order_dense_layer ##
    fst_order_dense_layer = Dense(4, activation='relu')(concat_dense_input)

    ####### linear concat ##########
    linear_part = concatenate([fst_order_dense_layer, fst_order_sparse_layer])

    #######dnn layer##########
    # Deep部分的输入好像只有FM的稀疏特征，没有稠密特征
    # concat_fm_embedding = concatenate([fm_embedding, fst_order_dense_layer], axis=-1)  # 加入稠密特征
    concat_fm_embedding = concatenate(fm_embedding, axis=-1)  # (None, 10*26)
    fc_layer = Dropout(0.2)(Activation(activation="relu")(BatchNormalization()(Dense(128)(concat_fm_embedding))))
    fc_layer = Dropout(0.2)(Activation(activation="relu")(BatchNormalization()(Dense(64)(fc_layer))))
    fc_layer = Dropout(0.2)(Activation(activation="relu")(BatchNormalization()(Dense(32)(fc_layer))))

    ######## output layer ##########
    # 注意：linear_part也要过sigmoid
    output_layer = concatenate([linear_part, snd_order_sparse_layer, fc_layer], axis=-1)  # (None, )
    output_layer = Dense(1, activation='sigmoid')(output_layer)

    model = Model(inputs=sparse_input + dense_input, outputs=output_layer)

    return model


model = deepfm_model(sparse_cols, dense_cols, x_train, x_valid)
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=["binary_crossentropy", tf.keras.metrics.AUC(name='auc')])

# 数据格式转换
train_sparse_x = [x_train[f].values for f in sparse_cols]
print('train_sparse_x = ', train_sparse_x)
train_dense_x = [x_train[f].values for f in dense_cols]
print('train_dense_x = ', train_dense_x)
print('train_sparse_x + train_dense_x = ', train_sparse_x + train_dense_x)

train_label = [y_train.values]

valid_sparse_x = [x_valid[f].values for f in sparse_cols]
valid_dense_x = [x_valid[f].values for f in dense_cols]
valid_label = [y_valid.values]

# 训练模型
from keras.callbacks import *

file_path = 'deepfm_model.h5'
earlystopping = EarlyStopping(monitor='val_auc', patience=3)
checkpoint = ModelCheckpoint(file_path,
                             save_weights_only=True,
                             verbose=1,
                             save_best_only=True)

callbacks_list = [earlystopping, checkpoint]
hist = model.fit(train_sparse_x + train_dense_x,
                 train_label,
                 batch_size=128,
                 epochs=20,
                 validation_data=(valid_sparse_x + valid_dense_x, valid_label),
                 callbacks=callbacks_list,
                 shuffle=False)

min_val_loss = np.min(hist.history['val_loss'])
max_val_auc = np.max(hist.history['val_auc'])
print('min_val_loss = ', min_val_loss)
print('max_val_auc = ', max_val_auc)
