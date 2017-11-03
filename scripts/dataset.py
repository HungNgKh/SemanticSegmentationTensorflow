import core.datamanager as dm
import numpy as np
from scipy import misc
import os, sys
import cv2


DATA_PATH = "/home/khanhhung/deeplearning/data/Segdata"
NUM_CLASS = 22



AUGMENTATION_METHODS = {
    'random_flip_horizontal': lambda x: __random_flip_horizontal(x),
    'random_flip_vertical': lambda x: __random_flip_vertical(x),
    'random_crop': lambda x: __random_crop(x, 224, 224)
}

def __random_flip_horizontal(batch):

    def random_flip_horizontal(image, label):
        rand = np.random.randint(2)
        if rand == 1:
            return image[:,::-1,:], label[:,::-1]
        else:
            return image, label

    datas = zip(batch.x, batch.y)
    datas = [random_flip_horizontal(x[0], x[1]) for x in datas]
    batch.x, batch.y = zip(*datas)
    return batch


def __random_flip_vertical(batch):

    def random_flip_vertical(image, label):
        rand = np.random.randint(2)
        if rand == 1:
            return image[::-1,:,:], label[::-1,:]
        else:
            return image, label

    datas = zip(batch.x, batch.y)
    datas = [random_flip_vertical(x[0], x[1]) for x in datas]
    batch.x, batch.y = zip(*datas)
    return batch


def __random_crop(batch, height, width):
    assert np.ndim(batch.x) == 4
    assert np.ndim(batch.y) == 3
    assert height <= np.shape(batch.x)[1]
    assert width <= np.shape(batch.x)[2]

    def random_crop(image, label, height, width):
        topleftx = np.random.randint(0, np.shape(image)[1] - width)
        toplefty = np.random.randint(0, np.shape(image)[0] - height)
        return image[toplefty:toplefty + height, topleftx:topleftx + width, :], label[toplefty:toplefty + height, topleftx:topleftx + width]

    datas = zip(batch.x, batch.y)
    datas = [random_crop(x[0], x[1], height, width) for x in datas]
    batch.x, batch.y = zip(*datas)
    return batch


class ColorLabelDecoder:

    def __init__(self):
        self.labelcolormap = dict()
        with open(DATA_PATH + "/labelcolor.dat", "r") as file:
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


class PascalVOCSegmentationDataSet(dm.RunTimeDataSet):

    def __init__(self, index_path):
        super(PascalVOCSegmentationDataSet, self).__init__()
        self.__image_path = DATA_PATH + '/Images/'
        self.__label_path = DATA_PATH + '/Labels/'
        self.__index_path = index_path


    def load(self):
        assert os.path.exists(self.__image_path)
        assert os.path.exists(self.__label_path)
        assert os.path.exists(self.__index_path)

        with open(self.__index_path, 'r') as file:
            self._index_list = [x.rstrip('\n') for x in file.readlines()]

        self.size = np.size(self._index_list, 0)


    def batch(self, batch_size):
        assert batch_size > 0
        images = []
        labels = []
        index_end = self._current_index + batch_size
        if (index_end >= self.size):
            index_end = self.size


        while self._current_index < index_end:
            index = self._index_list[self._current_index]
            img = cv2.imread(self.__image_path + index + '.jpg')
            label = np.loadtxt(self.__label_path + index + '.txt', dtype=np.uint8)
            label[label == NUM_CLASS] = 0
            images.append(img)
            labels.append(label)
            self._current_index += 1

        if(self._current_index >= self.size):
            self._current_index = 0
            return dm.Batch(images, labels), True
        else:
            return dm.Batch(images, labels), False


# class PascalVOCSegmentationDataSet(dm.LoadTimeDataSet):
#     def __init__(self, index_path):
#         super(PascalVOCSegmentationDataSet, self).__init__()
#         self.__image_path = DATA_PATH + '/Images/'
#         self.__label_path = DATA_PATH + '/Labels/'
#         self.__index_path = index_path
#
#
#     def load(self):
#         assert os.path.exists(self.__image_path)
#         assert os.path.exists(self.__label_path)
#         assert os.path.exists(self.__index_path)
#
#         with open(self.__index_path, 'r') as file:
#             self._index_list = [x.rstrip('\n') for x in file.readlines()]
#
#         self.size = 0
#
#         for x in self._index_list:
#             img = cv2.imread(self.__image_path + x + '.jpg')
#             label = np.loadtxt(self.__label_path + x + '.txt', dtype=np.uint8)
#             self.images.append(img)
#             self.labels.append(label)
#             self.size += 1
#             print x





if __name__ == "__main__":
    data = PascalVOCSegmentationDataSet(DATA_PATH + '/train.txt')
    data.load()

    data.shuffle()

    batch, _ = data.batch(5)

    method_list = [AUGMENTATION_METHODS['random_flip_horizontal'], AUGMENTATION_METHODS['random_flip_vertical'], AUGMENTATION_METHODS['random_crop']]
    img = batch.x[0]
    label = batch.y[0]

    cv2.imshow("img", img)
    cv2.waitKey(0)

    decoder = ColorLabelDecoder()
    cv2.imshow("img", decoder.decode(label))
    cv2.waitKey(0)

    batch = dm.data_augment(batch, method_list)
    img = batch.x[0]
    label = batch.y[0]

    cv2.imshow("img", img)
    cv2.waitKey(0)

    decoder = ColorLabelDecoder()
    cv2.imshow("img", decoder.decode(label))
    cv2.waitKey(0)

