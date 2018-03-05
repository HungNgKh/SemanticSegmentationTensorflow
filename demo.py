from __future__ import division
from PIL.Image import Image
from os.path import exists
import matplotlib.pyplot as plt
import tensorflow as tf
from scipy import misc
import numpy as np
import scipy.io
import sklearn
import matplotlib.image as mpimg
from PIL import Image
import cv2
from scripts.dataprocessing import pascalvoc
import scripts.dataset as dataset
import scripts.fcn8s_vgg16 as fcn8
import sqlite3


MODEL_PATH = "/home/khanhhung/deeplearning/SemanticSegmentation/data/progress/fcn8s_vgg16/training/"
MODEL_NAME = "fcn8s_vgg16"

IMG_PATH = "/home/khanhhung/deeplearning/SemanticSegmentation/data/assets/Images/1004973as.jpg"

if __name__ == "__main__":


    decoder = pascalvoc.ColorLabelDecoder()


    img = cv2.imread(IMG_PATH)
    # img /= 255
    config = tf.ConfigProto(
        device_count={'CPU': 1, 'GPU': 1},
        allow_soft_placement=True,
        log_device_placement=True
    )

    with tf.Session(config=config) as sess:
        saver = tf.train.import_meta_graph(MODEL_PATH + MODEL_NAME + '.meta')
        saver.restore(sess, tf.train.latest_checkpoint(MODEL_PATH))

        input = sess.graph.get_tensor_by_name('input:0')
        is_training = sess.graph.get_tensor_by_name('train_phase:0')

        predict = tf.squeeze(tf.argmax(sess.graph.get_tensor_by_name('upscore8_1:0'), 3))

        # print predict

        result = sess.run(predict, feed_dict={input: [img], is_training: False})

        # print np.max(result, 0)

        cv2.imshow('image', img)
        cv2.waitKey(0)

        cv2.imshow('result', decoder.decode(result))

        cv2.waitKey(0)
