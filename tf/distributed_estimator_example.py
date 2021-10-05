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
    f = feature_column.categorical_column_with_hash_bucket(feature_name, hash_bucket_size=hash_bucket_size, dtype=dtype)
    f1 = feature_column.embedding_column(f, 4)
    feature_layer = tf.keras.layers.DenseFeatures([f1])
    return feature_layer


feature_columns = [fc_column('device_ip', 100), fc_column('C1', 100, dtype=tf.int32)]


def input_fn(file_path):
    dataset = tf.data.experimental.make_csv_dataset(file_path,
                                                    batch_size=10,
                                                    column_names=columns,
                                                    label_name='click',
                                                    na_value="?",
                                                    num_epochs=1)
    dataset = dataset.shuffle(500)
    return dataset


# estimator = tf.estimator.LinearClassifier(feature_columns=feature_columns)
# estimator.train(input_fn=lambda: input_fn("avazu-ctr-prediction/train", 10), steps=2000)

# Checkpoints
# Need to talk three points:
# 1, checkpoints structure;
# 2, reload from a checkpoint as latest;
# 3, checkpoints must share the same network structure
checkpointing_config = tf.estimator.RunConfig(
    save_checkpoints_secs=30,  # Save checkpoints every 20 minutes.
    keep_checkpoint_max=5,  # Retain the 10 most recent checkpoints.
)


def model_fn(features, labels, mode, params):
    global_step = tf.train.get_global_step()
    device_ip = fc_transform('device_ip', 100)(features)
    C1 = fc_transform('C1', 100, dtype=tf.int32)(features)
    # C14 = fc_transform('C1', 100, dtype=tf.int32)(features)
    with tf.variable_scope("haha"):
        t1 = tf.keras.layers.Concatenate(axis=-1)([device_ip, C1])
        t2 = tf.keras.layers.Dense(4, activation='relu')(t1)
        logits = tf.keras.layers.Dense(1)(t2)
        predicted_logit = tf.nn.sigmoid(logits)
    predictions = {'probabilities': predicted_logit}
    # Predict Spec
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)
    # Eval Spec
    with tf.name_scope('loss'):
        cross_entropy = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.cast(labels, tf.float32), logits=tf.squeeze(logits)))
        tf.summary.scalar('loss', cross_entropy)
    with tf.name_scope('accuracy'):
        accuracy = tf.metrics.accuracy(labels=labels, predictions=predicted_logit, name='acc')
        tf.summary.scalar('accuracy', accuracy[1])
    with tf.name_scope('auc'):
        accuracy = tf.metrics.auc(labels=labels, predictions=predicted_logit, name='auc')
        tf.summary.scalar('auc', accuracy[1])
    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(mode=mode,
                                          loss=cross_entropy,
                                          eval_metric_ops={'accuracy/accuracy': accuracy},
                                          evaluation_hooks=None)
    # Train Spec
    optimizer = tf.train.AdamOptimizer()
    train_op = optimizer.minimize(cross_entropy, global_step=global_step)

    # Create a hook to print acc, loss & global step every 100 iter.
    train_hook_list = []
    train_tensors_log = {'accuracy': accuracy[1], 'loss': cross_entropy, 'global_step': global_step}
    train_hook_list.append(tf.train.LoggingTensorHook(tensors=train_tensors_log, every_n_iter=10))

    if mode == tf.estimator.ModeKeys.TRAIN:
        return tf.estimator.EstimatorSpec(mode=mode,
                                          loss=cross_entropy,
                                          train_op=train_op,
                                          training_hooks=train_hook_list)


import json
import os


def main(_):
    ps_hosts = FLAGS.ps_hosts.split(",")
    worker_hosts = FLAGS.worker_hosts.split(",")

    assert len(worker_hosts) >= 2
    # if FLAGS.job_name == 'worker' and FLAGS.task_index == 0:
    #     FLAGS.job_name = 'chief'
    # elif FLAGS.job_name == 'worker':
    #     FLAGS.task_index -= 1
    # # Create a cluster from the parameter server and worker hosts.
    # cluster_info = {
    #     'cluster': {
    #         "ps": ps_hosts,
    #         "worker": worker_hosts[1:],
    #         "chief": [worker_hosts[0]]
    #     },
    #     'task': {
    #         'type': FLAGS.job_name,
    #         'index': FLAGS.task_index
    #     }
    # }
    cluster_info = {
        'cluster': {
            "ps": ps_hosts,
            "worker": worker_hosts,
        },
        'task': {
            'type': FLAGS.job_name,
            'index': FLAGS.task_index
        }
    }
    print(cluster_info)
    os.environ['TF_CONFIG'] = json.dumps(cluster_info)
    os.environ['GRPC_VERBOSITY'] = 'DEBUG'

    # Strategy: MirroredStrategy, MultiWorkerMirroredStrategy, ParameterServerStrategy
    # https://fyubang.com/2019/07/08/distributed-training/
    strategy = tf.distribute.experimental.ParameterServerStrategy()
    config = tf.estimator.RunConfig(train_distribute=strategy)

    # tf.logging.set_verbosity(tf.logging.INFO)
    estimator = tf.estimator.Estimator(model_fn=model_fn, params={}, model_dir="./model_dir", config=config)
    tf.estimator.train_and_evaluate(
        estimator,
        train_spec=tf.estimator.TrainSpec(input_fn=lambda: input_fn("avazu-ctr-prediction/train")),
        eval_spec=tf.estimator.EvalSpec(input_fn=lambda: input_fn("avazu-ctr-prediction/train")))
    # metrics = estimator.evaluate(input_fn=lambda: input_fn("avazu-ctr-prediction/train", 10))
    # Homework: try estimator's train_and_evaluate function


if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser()
    parser.register("type", "bool", lambda v: v.lower() == "true")
    # Flags for defining the tf.train.ClusterSpec
    parser.add_argument("--ps_hosts", type=str, default="", help="Comma-separated list of hostname:port pairs")
    parser.add_argument("--worker_hosts", type=str, default="", help="Comma-separated list of hostname:port pairs")
    parser.add_argument("--job_name", type=str, default="", help="One of 'ps', 'worker'")
    # Flags for defining the tf.train.Server
    parser.add_argument("--task_index", type=int, default=0, help="Index of task within the job")
    parser.add_argument("--is_sync", type=bool, default=False, help="sync")
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
