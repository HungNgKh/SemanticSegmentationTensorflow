import tensorflow as tf
import numpy as np
import cv2

PASCAL_VOC_PATH = "/home/khanhhung/deeplearning/SemanticSegmentation/data/pascal_voc/"



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



class ColorLabelDecoder:

    def __init__(self):
        self.labelcolormap = dict()
        with open(PASCAL_VOC_PATH + "raw/labelcolor.dat", "r") as file:
            content = file.readlines()
            content = [x.rstrip('\n') for x in content]
            for x in content:
                values = x.split('\t')
                index = int(values[0])
                values = values[1].split(' ')
                self.labelcolormap[index] = (int(values[2]), int(values[1]), int(values[0]))
        self.colorlabelmap = {x: y for y,x in self.labelcolormap.iteritems()}


    def encode(self, image):
        shape = np.shape(image)
        en_img = np.zeros([shape[0], shape[1]])
        # en_img = [color_map_r[tuple(x)] for x in en_img]
        for i in range(shape[0]):
            for j in range(shape[1]):
                en_img[i][j] = self.colorlabelmap[tuple(image[i][j])]

        return en_img

    def decode(self, image):
        shape = np.shape(image)

        de_img = np.zeros([shape[0], shape[1], 3], 'uint8')
        for i in range(shape[0]):
            for j in range(shape[1]):
                de_img[i][j] = self.labelcolormap[image[i][j]]

        return de_img




def one_hot_decode(img, num_class):
    # shape = np.shape(img)
    # one_hot_img = np.zeros([shape[0], shape[1], num_class])
    one_hot_img = (np.arange(num_class) == img[:,:,None]).astype(int)
    return one_hot_img





def raw_to_tfrecord():

    def _bytes_feature(value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    def _int64_feature(value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

    raw_file = PASCAL_VOC_PATH + 'raw/train.txt'

    with open(raw_file, 'r') as file:
        index_list = [x.rstrip('\n') for x in file.readlines()]


    tfrecords_filename = PASCAL_VOC_PATH + 'tensorflow/train.tfrecords'

    writer = tf.python_io.TFRecordWriter(tfrecords_filename)

    for index in index_list:
        img = np.array(cv2.imread(PASCAL_VOC_PATH + 'raw/Images/' + index + '.jpg'), dtype=np.uint8)
        truth = np.loadtxt(PASCAL_VOC_PATH + 'raw/Labels/' + index + '.txt', dtype=np.uint8)

        # we need to store height and width to read and decode the raw serialized string later
        img_height = img.shape[0]
        img_width = img.shape[1]


        img_raw = img.tostring()
        truth_raw = truth.tostring()

        example = tf.train.Example(features=tf.train.Features(feature={
            'height': _int64_feature(img_height),
            'width': _int64_feature(img_width),
            'image_raw': _bytes_feature(img_raw),
            'mask_raw': _bytes_feature(truth_raw)}))

        writer.write(example.SerializeToString())
        print index

    writer.close()


if __name__ == "__main__":
    raw_to_tfrecord()