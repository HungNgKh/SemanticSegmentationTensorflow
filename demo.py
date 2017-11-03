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
import scripts.dataset as dataset
import scripts.fcn8 as fcn8
import sqlite3

DATA_PATH = "/home/khanhhung/deeplearning/data/Segdata"

def color_map(N=256, normalized=False):
    def bitget(byteval, idx):
        return ((byteval & (1 << idx)) != 0)

    dtype = 'float32' if normalized else 'uint8'
    cmap = np.zeros((N, 3), dtype=dtype)
    for i in range(N):
        r = g = b = 0
        c = i
        for j in range(8):
            r = r | (bitget(c, 0) << 7-j)
            g = g | (bitget(c, 1) << 7-j)
            b = b | (bitget(c, 2) << 7-j)
            c = c >> 3

        cmap[i] = np.array([r, g, b])

    cmap = cmap/255 if normalized else cmap
    return cmap


def color_map_viz():
    labels = ['background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'scripts', 'tvmonitor', 'void']
    nclasses = 21
    row_size = 50
    col_size = 500
    cmap = color_map()
    array = np.empty((row_size*(nclasses+1), col_size, cmap.shape[1]), dtype=cmap.dtype)
    for i in range(nclasses):
        array[i*row_size:i*row_size+row_size, :] = cmap[i]
    array[nclasses*row_size:nclasses*row_size+row_size, :] = cmap[-1]

    file = open("../data/Segdata/labelcolor.dat", "w")
    for i in range(nclasses):
        file.write(str(i))
        file.write('\t')
        file.write(str(cmap[i][0]))
        file.write(' ')
        file.write(str(cmap[i][1]))
        file.write(' ')
        file.write(str(cmap[i][2]))
        file.write('\n')

    file.write(str(nclasses))
    file.write('\t')
    file.write(str(cmap[-1][0]))
    file.write(' ')
    file.write(str(cmap[-1][1]))
    file.write(' ')
    file.write(str(cmap[-1][2]))
    file.write('\n')



def loadcolormap():
    colordict = dict()
    with open("../data/Segdata/labelcolor.dat", "r") as file:
        content = file.readlines()
        content = [x.rstrip('\n') for x in content]
        for x in content:
            values = x.split('\t')
            index = int(values[0])
            values = values[1].split(' ')
            colordict[index] = (int(values[2]), int(values[1]), int(values[0]))

    return colordict

def encode(img, color_map_r):
    shape = np.shape(img)
    en_img = np.zeros([shape[0], shape[1]])
    # en_img = [color_map_r[tuple(x)] for x in en_img]
    for i in range(shape[0]):
        for j in range(shape[1]):
            en_img[i][j] =  color_map_r[tuple(img[i][j])]

    return en_img

def decode(img, color_map):
    shape = np.shape(img)

    de_img = np.zeros([shape[0], shape[1], 3], 'uint8')
    for i in range(shape[0]):
        for j in range(shape[1]):
            de_img[i][j] =  color_map[img[i][j]]

    return de_img







if __name__ == "__main__":
    # label = cv2.imread(dataset.DATA_PATH + '/JPEGImages/2007_002281.jpg')
    #
    # decoder = dataset.ColorLabelDecoder()
    # # cv2.f
    #
    # cv2.imshow('img', label)
    # cv2.waitKey(0)
    #
    # resized = cv2.resize(label, (256, 256), interpolation=cv2.INTER_NEAREST)
    # # resized.astype(np.uint8)
    # cv2.imshow('img', resized)
    # cv2.waitKey(0)
    performance = scipy.io.loadmat(fcn8.PERFORMANCE_PROGRESS_FILE)
    print performance
    # cv2.imshow('img', decoder.decode(decoder.encode(resized)))
    # cv2.waitKey(0)
    #
    # print np.array_equal(resized, decoder.decode(decoder.encode(resized)))
