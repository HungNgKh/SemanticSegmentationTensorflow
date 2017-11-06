import unittest
import tensorflow as tf
import numpy as np
from scripts.core.datamanager import Batch
from scripts.dataprocessing import pascalvoc
from scripts.core.datamanager import TensorflowDataset

class TestTfrecordDataset(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super(TestTfrecordDataset, self).__init__(*args, **kwargs)
        self.sess = tf.Session()


    def test_output_shape(self):
        dataset = TensorflowDataset(path=pascalvoc.PASCAL_VOC_PATH + 'tensorflow/train.tfrecords', batch_size=5, image_shape=[256, 256, 3], truth_shape=[256, 256], epoch_size= 1464, numthread=4)

        # self.sess.run(tf.local_variables_initializer())
        # self.sess.run(tf.global_variables_initializer())
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord, sess=self.sess)

        batch = dataset.batch(self.sess)

        img_shape = np.shape(batch.x)
        mask_shape = np.shape(batch.y)

        np.testing.assert_equal(img_shape, [5, 256, 256, 3])
        np.testing.assert_equal(mask_shape, [5, 256, 256])

        coord.request_stop()
        coord.join(threads)


    def test_output_labels(self):
        dataset = TensorflowDataset(path=pascalvoc.PASCAL_VOC_PATH + 'tensorflow/train.tfrecords', batch_size=5,
                                    image_shape=[256, 256, 3], truth_shape=[256, 256], epoch_size= 1464, numthread=4)

        # self.sess.run(tf.local_variables_initializer())
        # self.sess.run(tf.global_variables_initializer())
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord, sess=self.sess)

        batch = dataset.batch(self.sess)

        decoder = pascalvoc.ColorLabelDecoder()
        for i in range(5):
            np.testing.assert_equal(batch.y[i], decoder.encode(decoder.decode(batch.y[i])))

        coord.request_stop()
        coord.join(threads)

