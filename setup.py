import tensorflow as tf
import scripts.fcn8s as fcn8
import tensorflow.contrib.layers as layer


# with tf.variable_scope("var"):
#     v1 = tf.get_variable('v1', [2, 2], initializer=layer.xavier_initializer())
# with tf.variable_scope("foo"):
#     v1 = tf.get_variable('v1', [2, 2], initializer=layer.xavier_initializer())

def sum():
    v1 = tf.get_default_graph().get_tensor_by_name('var/v1:0')
    v2 = tf.get_default_graph().get_tensor_by_name('foo/v1:0')
    op = tf.add(v1, v2, 'sum')
    # op2 = tf.add(op, v2, 'sum2')
    # op3 = tf.add(op2, v2, 'sum3')

with tf.Session() as sess:
    saver = tf.train.import_meta_graph(fcn8.MODEL_PATH + fcn8.MODEL_NAME + '.meta')
    saver.restore(sess, tf.train.latest_checkpoint(fcn8.MODEL_PATH))


    # sess.run(tf.global_variables_initializer())


    # a = tf.get_variable('v3', [2, 2], initializer=layer.xavier_initializer())
    # a = tf.placeholder(tf.float32, shape=[2], name='pl')
    print tf.global_variables()
    print sess.graph.get_tensor_by_name("pl:0")

    # sess.run(tf.global_variables_initializer())
    # saver = tf.train.Saver()
    saver.save(sess,fcn8.MODEL_PATH + fcn8.MODEL_NAME, write_meta_graph=False)
