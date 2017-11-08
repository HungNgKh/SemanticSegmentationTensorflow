import unittest
import scripts.dataset as dataset
import tensorflow as tf
import numpy as np
from scripts.dataprocessing import pascalvoc

class TestPascalVocTfrecord(unittest.TestCase):


    def __init__(self, *args, **kwargs):
        super(TestPascalVocTfrecord, self).__init__(*args, **kwargs)
        raw_dataset = dataset.PascalVOCSegmentationDataSet(batch_size=5, index_path=pascalvoc.PASCAL_VOC_PATH + '/raw/train.txt')
        raw_dataset.load()
        self.raw_data, _ = raw_dataset.batch()
        self.record = tf.python_io.tf_record_iterator(path=pascalvoc.PASCAL_VOC_PATH + '/tensorflow/train.tfrecords')


    def test_decode_image(self):
        images = self.raw_data.x
        i = 0
        for string in self.record:
            example = tf.train.Example()
            example.ParseFromString(string)

            height = int(example.features.feature['height'].int64_list.value[0])

            width = int(example.features.feature['width'].int64_list.value[0])

            img_string = example.features.feature['image_raw'].bytes_list.value[0]

            decoded_img = np.fromstring(img_string, dtype=np.uint8)
            decoded_img = np.reshape(decoded_img, (height, width, -1))

            np.testing.assert_equal(images[i], decoded_img)
            i += 1

            if i >= np.size(images, 0):
                break


    def test_decode_truth(self):
        truths = self.raw_data.y
        i = 0
        for string in self.record:
            example = tf.train.Example()
            example.ParseFromString(string)

            height = int(example.features.feature['height'].int64_list.value[0])

            width = int(example.features.feature['width'].int64_list.value[0])

            img_string = example.features.feature['mask_raw'].bytes_list.value[0]

            decoded_truth = np.fromstring(img_string, dtype=np.uint8)
            decoded_truth = np.reshape(decoded_truth, (height, width))

            np.testing.assert_equal(truths[i], decoded_truth)
            i += 1
            if i >= np.size(truths, 0):
                break


    def test_color_label_decoder(self):
        decoder = pascalvoc.ColorLabelDecoder()
        test_iter = 5
        i = 0
        for string in self.record:
            example = tf.train.Example()
            example.ParseFromString(string)

            height = int(example.features.feature['height'].int64_list.value[0])

            width = int(example.features.feature['width'].int64_list.value[0])

            img_string = example.features.feature['mask_raw'].bytes_list.value[0]

            decoded_truth = np.fromstring(img_string, dtype=np.uint8)
            decoded_truth = np.reshape(decoded_truth, (height, width))

            np.testing.assert_equal(decoded_truth, decoder.encode(decoder.decode(decoded_truth)))
            i += 1
            if i >= test_iter:
                break


    def test_dataset_difference(self):
        train_record = tf.python_io.tf_record_iterator(path=pascalvoc.PASCAL_VOC_PATH + '/tensorflow/train.tfrecords')
        val_record = tf.python_io.tf_record_iterator(path=pascalvoc.PASCAL_VOC_PATH + '/tensorflow/validation.tfrecords')
        test_iter = 5
        i = 0
        for x,y in zip(train_record, val_record):

            # get train iamge
            example = tf.train.Example()
            example.ParseFromString(x)

            height = int(example.features.feature['height'].int64_list.value[0])

            width = int(example.features.feature['width'].int64_list.value[0])

            img_string = example.features.feature['mask_raw'].bytes_list.value[0]

            decoded_x = np.fromstring(img_string, dtype=np.uint8)
            decoded_x = np.reshape(decoded_x, (height, width))

            # get y image
            example = tf.train.Example()
            example.ParseFromString(y)

            height = int(example.features.feature['height'].int64_list.value[0])

            width = int(example.features.feature['width'].int64_list.value[0])

            img_string = example.features.feature['mask_raw'].bytes_list.value[0]

            decoded_y = np.fromstring(img_string, dtype=np.uint8)
            decoded_y = np.reshape(decoded_y, (height, width))

            self.assertTrue(np.any(np.not_equal(decoded_x, decoded_y)))
            i += 1
            if i >= test_iter:
                break