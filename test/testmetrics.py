import unittest
import scripts.core.utility as util
import tensorflow as tf

class TestMetrics(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super(TestMetrics, self).__init__(*args, **kwargs)
        self.sess = tf.Session()


    def test_accuracy(self):
        x = tf.placeholder(tf.int64, [5])
        y = tf.placeholder(tf.int64, [5])

        metric = util.MetricsCalculator(self.sess, labels=y, predicts=x, num_classes=2, name='metrics')
        self.sess.run(tf.local_variables_initializer())
        metric.reset()

        self.sess.run(metric.accuracy_op, feed_dict={x: [1, 0, 0, 0, 0], y: [1, 0, 0, 0, 1]})
        self.sess.run(metric.accuracy_op, feed_dict={x: [1, 0, 0, 0, 0],y: [0, 1, 1, 1, 1]})

        self.assertAlmostEqual(self.sess.run(metric.accuracy), 0.4, delta=1e-3)


    def test_mean_accuracy(self):
        x = tf.placeholder(tf.int64, [5])
        y = tf.placeholder(tf.int64, [5])

        metric = util.MetricsCalculator(self.sess, labels=y, predicts=x, num_classes=2, name='metrics')
        self.sess.run(tf.local_variables_initializer())
        metric.reset()

        self.sess.run(metric.mean_accuracy_op, feed_dict={x: [1, 0, 0, 0, 0], y: [1, 0, 0, 0, 1]})
        self.sess.run(metric.mean_accuracy_op, feed_dict={x: [1, 0, 0, 0, 0],y: [0, 1, 1, 1, 1]})

        self.assertAlmostEqual(self.sess.run(metric.mean_accuracy), 0.4583, delta=1e-3)


    def test_meanUI(self):
        x = tf.placeholder(tf.int64, [5])
        y = tf.placeholder(tf.int64, [5])

        metric = util.MetricsCalculator(self.sess, labels=y, predicts=x, num_classes=2, name='metrics')
        self.sess.run(tf.local_variables_initializer())
        metric.reset()

        self.sess.run(metric.meanIU_op, feed_dict={x: [1, 0, 0, 0, 0], y: [1, 0, 0, 0, 1]})
        self.sess.run(metric.meanIU_op, feed_dict={x: [1, 0, 0, 0, 0], y: [0, 1, 1, 1, 1]})

        self.assertAlmostEqual(self.sess.run(metric.meanIU), 0.238, delta=1e-3)


    def test_reset(self):
        x = tf.placeholder(tf.int64, [5])
        y = tf.placeholder(tf.int64, [5])

        metric = util.MetricsCalculator(self.sess, labels=y, predicts=x, num_classes=2, name='metrics')
        self.sess.run(tf.local_variables_initializer())
        metric.reset()

        self.sess.run(metric.accuracy_op, feed_dict={x: [1, 0, 0, 0, 0], y: [1, 0, 0, 0, 1]})
        self.assertAlmostEqual(self.sess.run(metric.accuracy), 0.8, delta=1e-3)

        metric.reset()

        self.sess.run(metric.accuracy_op, feed_dict={x: [1, 0, 0, 0, 0], y: [1, 0, 0, 0, 1]})
        self.assertAlmostEqual(self.sess.run(metric.accuracy), 0.8, delta=1e-3)
