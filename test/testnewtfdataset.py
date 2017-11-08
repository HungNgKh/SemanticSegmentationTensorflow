from __future__ import division
import unittest
import tensorflow as tf
import numpy as np
# from scripts.core.datamanager import Batch
from scripts.dataprocessing import pascalvoc
from scripts.core.datamanager import TensorflowDataset
from scripts.core.datamanager import NewTFDataset
import cv2



class TestNewTFDataset(unittest.TestCase):


    def __init__(self, *args, **kwargs):
        super(TestNewTFDataset, self).__init__(*args, **kwargs)
        self.sess = tf.Session()


    def test_output_shape(self):
        dataset = NewTFDataset(path=pascalvoc.PASCAL_VOC_PATH + 'tensorflow/train.tfrecords', batch_size=5, image_shape=[256, 256, 3], truth_shape=[256, 256], epoch_size= 1464, sess=self.sess)

        # self.sess.run(tf.local_variables_initializer())
        # self.sess.run(tf.global_variables_initializer())
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord, sess=self.sess)

        batch  = dataset.batch(self.sess)

        img_shape = np.shape(batch.x)
        mask_shape = np.shape(batch.y)

        np.testing.assert_equal(img_shape, [5, 256, 256, 3])
        np.testing.assert_equal(mask_shape, [5, 256, 256])

        coord.request_stop()
        coord.join(threads)


    def test_output_labels(self):
        dataset = NewTFDataset(path=pascalvoc.PASCAL_VOC_PATH + 'tensorflow/train.tfrecords', batch_size=5,
                                    image_shape=[256, 256, 3], truth_shape=[256, 256], epoch_size= 1464, sess=self.sess)

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


    def test_epoch_size(self):
        dataset = NewTFDataset(path=pascalvoc.PASCAL_VOC_PATH + 'tensorflow/train.tfrecords', batch_size=5,
                                    image_shape=[256, 256, 3], truth_shape=[256, 256], epoch_size=1464, sess=self.sess)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord, sess=self.sess)
        # self.sess.run(tf.global_variables_initializer())
        # self.sess.run()

        for i in range(292):
            batch = dataset.batch(self.sess)
            self.assertEqual(np.size(batch.x, 0), 5)
            self.assertEqual(np.size(batch.y, 0), 5)

        batch = dataset.batch(self.sess)
        self.assertEqual(np.size(batch.x, 0), 4)
        self.assertEqual(np.size(batch.y, 0), 4)

        batch = dataset.batch(self.sess)

        self.assertEqual(batch, None)

        coord.request_stop()
        coord.join(threads)



    def test__reset(self):
        dataset = NewTFDataset(path=pascalvoc.PASCAL_VOC_PATH + 'tensorflow/train.tfrecords', batch_size=5,
                               image_shape=[256, 256, 3], truth_shape=[256, 256], epoch_size=1464, sess=self.sess)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord, sess=self.sess)
        # self.sess.run(tf.global_variables_initializer())

        for i in range(293):
            batch = dataset.batch(self.sess)
            self.assertTrue(batch is not None)

        batch = dataset.batch(self.sess)
        self.assertFalse(batch is not None)

        dataset.reset(self.sess)
        batch = dataset.batch(self.sess)
        self.assertTrue(batch is not None)


        coord.request_stop()
        coord.join(threads)



    def test_shuffle(self):
        dataset = NewTFDataset(path=pascalvoc.PASCAL_VOC_PATH + 'tensorflow/train.tfrecords', batch_size=5,
                               image_shape=[256, 256, 3], truth_shape=[256, 256], epoch_size=1464, sess=self.sess)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord, sess=self.sess)
        # self.sess.run(tf.global_variables_initializer())

        batch1 = dataset.batch(self.sess)

        for i in range(292):
            dataset.batch(self.sess)

        dataset.reset(self.sess)


        batch2 = dataset.batch(self.sess)
        #
        self.assertFalse(np.array_equal(batch1.x, batch2.x))
        self.assertFalse(np.array_equal(batch1.y, batch2.y))

        cv2.imshow("img", batch1.x[0])
        cv2.waitKey(0)

        cv2.imshow("img", batch2.x[0])
        cv2.waitKey(0)

        coord.request_stop()
        coord.join(threads)