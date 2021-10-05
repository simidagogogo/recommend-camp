import tensorflow as tf
from tensorflow import feature_column
from tensorflow.keras import layers
# from tensorflow.python.feature_column import feature_column_v2 as fc

columns = [
    'id',
    'click',
    'hour',
    'C1',
    'banner_pos',
    'site_id',
    'site_domain',
    'site_category',
    'app_id',
    'app_domain',
    'app_category',
    'device_id',
    'device_ip',
    'device_model',
    'device_type',
    'device_conn_type',
    'C14',
    'C15',
    'C16',
    'C17',
    'C18',
    'C19',
    'C20',
    'C21',
]


# Step I, read data
# Option I, read data into memory: pandas dataframe
# Option II, read data by batch
def get_dataset():
    dataset = tf.data.experimental.make_csv_dataset("avazu-ctr-prediction/train",
                                                    batch_size=64,
                                                    column_names=columns,
                                                    label_name='click',
                                                    num_epochs=1)
    return dataset


# Step II, consume data
raw_train_data = get_dataset()
k = raw_train_data.make_initializable_iterator()
features, labels = k.get_next()


# Step III, use feature column to do feature transformation
def fc_transform(feature_name, hash_bucket_size, dtype=tf.string):
    f = feature_column.categorical_column_with_hash_bucket(feature_name, hash_bucket_size=hash_bucket_size, dtype=dtype)
    f1 = feature_column.embedding_column(f, 4)
    feature_layer = layers.DenseFeatures([f1])
    return feature_layer


# step IV, network body
device_ip = fc_transform('device_ip', 100, tf.string)(features)
C1 = fc_transform('C1', 100, dtype=tf.int32)(features)
with tf.variable_scope("haha"):
    t1 = tf.keras.layers.Concatenate(axis=-1)([device_ip, C1])
    t2 = tf.keras.layers.Dense(8, activation='relu')(t1)
    t3 = tf.keras.layers.Dense(8, activation='relu')(t2)
    t4 = tf.keras.layers.Dense(1)(t3)

print(t1)

# Step V, loss and optimizer
loss_op = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.reshape(tf.cast(labels, tf.float32), [-1, 1]),
                                                  logits=tf.squeeze(t3))
update_op = tf.train.AdamOptimizer().minimize(loss_op)

# Step VI, tensorboard
tf.summary.scalar('loss', tf.reduce_mean(loss_op))
tf.summary.scalar('auc', tf.metrics.auc(labels, t3))
# 看看t3是什么？
merged_op = tf.summary.merge_all() # summary可能有好多，需要合在一起
eval_writer = tf.summary.FileWriter('./model_dir/eval')

# Step VII, run with session
if __name__ == '__main__':
    with tf.Session() as sess:
        sess.run(k.initializer)
        sess.run(tf.global_variables_initializer())
        for i in range(1000):
            if i % 10 == 0:
                merged, loss_value = sess.run([merged_op, tf.reduce_sum(loss_op)])
                eval_writer.add_summary(merged, i)
            else:
                sess.run(update_op)
