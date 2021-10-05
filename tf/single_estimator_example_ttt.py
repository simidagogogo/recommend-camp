import tensorflow as tf
from tensorflow import feature_column

# from tensorflow.python.feature_column import feature_column_v2 as fc

# tf.enable_eager_execution()

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


def fc_column(feature_name, hash_bucket_size, dtype=tf.string):
    f = feature_column.categorical_column_with_hash_bucket(feature_name, hash_bucket_size=hash_bucket_size, dtype=dtype)
    f1 = feature_column.embedding_column(f, 10)
    return f1


def fc_transform(feature_name, hash_bucket_size, dtype=tf.string):
    feature_layer = tf.keras.layers.DenseFeatures([fc_column(feature_name, hash_bucket_size, dtype)])
    return feature_layer


feature_columns = [fc_column('device_ip', 100), fc_column('C1', 100, dtype=tf.int32)]


# Tensorflow have three levels of API:
# Low Level API, tf.reduce_sum, tf.matmul
# Mid Level API, layers, tf.keras.layers. Dense, Concatenate. Customize a keras layers
# TODO: Customize a keras layers
# High Level API, Estimator. Session and Graph.
# Advantages: without session, you can focus on model logic
# Disadvantages; you have less control with the model, Hook
# TODO: Hook


def input_fn(file_path):
    dataset = tf.data.experimental.make_csv_dataset(file_path,
                                                    batch_size=10,
                                                    column_names=columns,
                                                    label_name='click',
                                                    na_value="?",
                                                    num_epochs=1)
    dataset = dataset.shuffle(500)
    return dataset


tf.logging.set_verbosity(tf.logging.INFO)


# estimator = tf.estimator.LinearClassifier(feature_columns=feature_columns)
# estimator.train(input_fn=lambda: input_fn("avazu-ctr-prediction/train"), max_steps=10000)


# model_fn(features: dict, labels, mode: Three modes, params={'optimizer': 'ftr'})
# model_dir="./model_dir" event/checkpoint
# config


def model_fn(features, labels, mode, params):
    global_step = tf.train.get_or_create_global_step()
    device_ip = fc_transform('device_ip', 100)(features)
    C1 = fc_transform('C1', 100, dtype=tf.int32)(features)
    with tf.variable_scope("ctr"):
        t1 = tf.keras.layers.Concatenate(axis=-1)([device_ip, C1])
        t2 = tf.keras.layers.Dense(4, activation='relu')(t1)
        ctr_logits = tf.keras.layers.Dense(1)(t2)
    with tf.variable_scope("cvr"):
        t3 = tf.keras.layers.Concatenate(axis=-1)([device_ip, C1])
        t4 = tf.keras.layers.Dense(4, activation='relu')(t1)
        cvr_logits = tf.keras.layers.Dense(1)(t4)

    ctr_predicted_logit = tf.nn.sigmoid(ctr_logits)
    cvr_predicted_logit = tf.nn.sigmoid(cvr_logits)

    # Three Modes (Train, Eval, Predict)
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions={"cvr": cvr_predicted_logit})
    ctr_cross_entropy = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.cast(labels[:, 0], tf.float32), logits=tf.squeeze(ctr_logits)))
    ctcvr_cross_entropy = tf.keras.backend.binary_crossentropy(ctr_predicted_logit*cvr_predicted_logit, tf.cast(labels[:, 1], tf.float32))
    loss = ctr_cross_entropy + ctcvr_cross_entropy
    tf.summary.scalar('loss', cross_entropy)
    accuracy = tf.metrics.accuracy(labels=labels, predictions=predicted_logit, name='acc')
    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(mode=mode, loss=cross_entropy, eval_metric_ops={'accuracy': accuracy},
                                          evaluation_hooks=None)
    if params['optimizer'] == 'adadelta':
        optimizer = tf.train.AdadeltaOptimizer()
    else:
        optimizer = tf.train.AdamOptimizer()
    train_op = optimizer.minimize(loss, global_step=global_step)
    if mode == tf.estimator.ModeKeys.TRAIN:
        return tf.estimator.EstimatorSpec(mode=mode, loss=cross_entropy, train_op=train_op)


estimator = tf.estimator.Estimator(model_fn=model_fn, model_dir="./model_dir",
                                   config=tf.estimator.RunConfig(save_checkpoints_secs=5, keep_checkpoint_max=5),
                                   params={'optimizer': 'adadelta'})
# estimator.train(input_fn=lambda: input_fn("avazu-ctr-prediction/train"), max_steps=100000)
# Event: Tensorboard
# Checkpoint: params, gradients
# SavedModel: params
tf.estimator.train_and_evaluate(estimator,
                                train_spec=tf.estimator.TrainSpec(
                                    input_fn=lambda: input_fn("avazu-ctr-prediction/train")),
                                eval_spec=tf.estimator.EvalSpec(input_fn=lambda: input_fn("avazu-ctr-prediction/train"),
                                                                throttle_secs=10))

# 1. 一般情况下, 使用 Adadelta OR Adam SGD->ADAM
# 2. 在对于稀疏性要求比较高的情况下, 建议 FTRL. GroupFTRL
