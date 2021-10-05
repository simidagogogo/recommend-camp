import tensorflow as tf
from tensorflow import feature_column
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
                                                    batch_size=8,
                                                    column_names=columns,
                                                    label_name='click',
                                                    num_epochs=1)
    return dataset


# Step III, use feature column to do feature transformation
def fc_transform(feature_name, hash_bucket_size, dtype=tf.string):
    f = feature_column.categorical_column_with_hash_bucket(feature_name, hash_bucket_size=hash_bucket_size, dtype=dtype)
    f1 = feature_column.embedding_column(f, 4)
    feature_layer = tf.keras.layers.DenseFeatures([f1])
    return feature_layer


# step IV, network body
def network():
    # Step II, consume data
    raw_train_data = get_dataset()
    # diff between make_initializable_iterator and make_one_shot_iterator: the latter does not need initialization
    # but the former can be used to initialize different files
    k = raw_train_data.make_initializable_iterator()
    features, labels = k.get_next()
    device_ip = fc_transform('device_ip', 100)(features)
    C1 = fc_transform('C1', 100, dtype=tf.int32)(features)
    with tf.variable_scope("haha"):
        t1 = tf.keras.layers.Concatenate(axis=-1)([device_ip, C1])
        t2 = tf.keras.layers.Dense(4, activation='relu')(t1)
        t3 = tf.keras.layers.Dense(1)(t2)

    # Step V, loss and optimizer
    loss_op = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.reshape(tf.cast(labels, tf.float32), [-1, 1]),
                                                      logits=tf.squeeze(t3))
    return k, loss_op


# Monitored Training Session
"""
https://zhuanlan.zhihu.com/p/91608555
https://zhuanlan.zhihu.com/p/35083779

创建完session之后，再包装一下返回最终的MonitoredSession类，

一个完整的monitored session在创建时间内可做的事情（按顺序）：

为每个hook调用hook.begin()
调用scaffold.finalize()完成graph
创建session
为模型参数做初始化 ，通过Scaffold
如果存在checkpoint则根据checkpoint restore参数
发布runners队列
调用hook.after_create_session()函数
当run函数调用时，monitored session做的事情：

调用hook.before_run()
调用TensorFlow中的 `session.run()` with merged fetches and feed_dict
调用hook.after_run()
返回session.run()的结果
如果发生AbortedError或者UnavailableError，则在再次执行run()之前恢复或者重新初始化会话
当close()函数调用时，monitored session做的事情：

调用 hook.end()
关闭queue runners 和session
如果所有的输入数据被消耗完，抛出OutOfRange异常。
"""


def main(_):
    ps_hosts = FLAGS.ps_hosts.split(",")
    worker_hosts = FLAGS.worker_hosts.split(",")

    # Create a cluster from the parameter server and worker hosts.
    cluster = tf.train.ClusterSpec({"ps": ps_hosts, "worker": worker_hosts})

    # Create and start a server for the local task.
    server = tf.train.Server(cluster, job_name=FLAGS.job_name, task_index=FLAGS.task_index)

    if FLAGS.job_name == "ps":
        server.join() # ps阻断自己，等worker起来之后，再一起工作
    elif FLAGS.job_name == "worker":

        # Assigns ops to the local worker by default.
        with tf.device(
                tf.train.replica_device_setter(worker_device="/job:worker/task:%d" % FLAGS.task_index,
                                               cluster=cluster)):

            hooks = []
            # The StopAtStepHook handles stopping after running given steps.
            hooks.append(tf.train.StopAtStepHook(last_step=10000000))

            # Build model...
            k, loss_op = network()
            global_step = tf.train.get_or_create_global_step()
            optimizer = tf.train.AdamOptimizer()

            # if FLAGS.is_sync:
            # asynchronous training
            # use tf.train.SyncReplicasOptimizer wrap optimizer
            # ref: https://www.tensorflow.org/api_docs/python/tf/train/SyncReplicasOptimizer
            optimizer = tf.train.SyncReplicasOptimizer(optimizer,
                                                       replicas_to_aggregate=2,
                                                       total_num_replicas=2)
            # create the hook which handles initialization and queues
            hooks.append(optimizer.make_session_run_hook((FLAGS.task_index == 0)))

            update_op = optimizer.minimize(loss_op, global_step=global_step)

            # Step VI, tensorboard
            tf.summary.scalar('loss', tf.reduce_mean(loss_op))
            merged_op = tf.summary.merge_all()

        scaffold = tf.train.Scaffold(local_init_op=tf.group(k.initializer, tf.local_variables_initializer()),
                                     summary_op=merged_op)
        # summarySaverHook = tf.train.SummarySaverHook(save_steps=100, output_dir="./model_dir", scaffold=scaffold)
        # hooks.append(summarySaverHook)

        # The MonitoredTrainingSession takes care of session initialization,
        # restoring from a checkpoint, saving to a checkpoint, and closing when done
        # or an error occurs.

        # Question: where does the writer come from? (hint: from tf.train.SummarySaverHook)
        # Also StepCounterHook (look at tensorboard, you can see the global_step/sec), CheckpointSaverHook
        with tf.train.MonitoredTrainingSession(master=server.target,
                                               is_chief=(FLAGS.task_index == 0),
                                               checkpoint_dir="./model_dir",
                                               hooks=hooks,
                                               scaffold=scaffold) as sess:
            while not sess.should_stop():
                # Run a training step asynchronously.
                # See /api_docs/python/tf/train/SyncReplicasOptimizer"
                # mon_sess.run handles AbortedError in case of preempted PS.
                # sess.run(tf.global_variables_initializer())
                merged, _, loss_value, step = sess.run([merged_op, update_op, loss_op, global_step])
                print(step)
                # if step % 1000 == 0:
                #     eval_writer.add_summary(merged, step)


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

# 通过在一个机器上开多个端口处理
# python lesson4/distributed_session_example.py \
#      --ps_hosts=localhost:2222,localhost:2223 \
#      --worker_hosts=localhost:2224,localhost:2225 \
#      --job_name=ps --task_index=0
