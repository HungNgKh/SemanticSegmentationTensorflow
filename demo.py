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


MODEL_PATH = "/home/khanhhung/deeplearning/SemanticSegmentation/data/progress/fcn8s/best/"
MODEL_NAME = "fcn8s"

IMG_PATH = "/home/khanhhung/deeplearning/SemanticSegmentation/data/assets/Images/2007_001763.jpg"

if __name__ == "__main__":


    decoder = pascalvoc.ColorLabelDecoder()


    img = cv2.imread(IMG_PATH)
    img /= 255
    with tf.Session() as sess:
        saver = tf.train.import_meta_graph(MODEL_PATH + MODEL_NAME + '.meta')
        saver.restore(sess, tf.train.latest_checkpoint(MODEL_PATH))

        input = sess.graph.get_tensor_by_name('input:0')
        is_training = sess.graph.get_tensor_by_name('is_training:0')
        predict = tf.squeeze(tf.argmax(sess.graph.get_tensor_by_name('deconv_3:0'), 3))
        # print predict

        result = sess.run(predict, feed_dict={input: [img], is_training: False})

        # print np.max(result, 0)

        cv2.imshow('image', img)
        cv2.waitKey(0)

        cv2.imshow('result', decoder.decode(result))

        cv2.waitKey(0)
