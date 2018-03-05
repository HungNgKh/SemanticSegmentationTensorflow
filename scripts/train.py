from __future__ import print_function, division
import core.utility as util
import fcn8s_vgg16
import scipy.io
import tensorflow as tf
import dataset
import os, cv2
import numpy as np
from dataprocessing import pascalvoc



BATCH_SIZE = 5
EPOCH_NUM = 2000

START_MOMENTUM = 0.5
MAX_MOMENTUM = 0.9
MOMENTUM_INCREASE = 0.05
MOMENTUM_UP_EPOCH_STEP = 50


def better_performance(result):
    assert os.path.exists(fcn8s_vgg16.PERFORMANCE_PROGRESS_FILE)

    assert result['pixel_accuracy'] >= 0
    assert result['pixel_accuracy'] <= 1

    assert result['mean_accuracy'] >= 0
    assert result['mean_accuracy'] <= 1

    assert result['meanIU'] >= 0
    assert result['meanIU'] <= 1

    assert result['loss'] >= 0

    performance = scipy.io.loadmat(fcn8s_vgg16.PERFORMANCE_PROGRESS_FILE)
    max_acc = -1

    if np.size(performance['meanIU']) > 0:
        max_acc = np.max(performance['meanIU'])

    performance['pixel_accuracy'] = np.append(performance['pixel_accuracy'], result['pixel_accuracy'])
    performance['mean_accuracy'] = np.append(performance['mean_accuracy'], result['mean_accuracy'])
    performance['meanIU'] = np.append(performance['meanIU'], result['meanIU'])
    performance['loss'] = np.append(performance['loss'], result['loss'])
    scipy.io.savemat(fcn8s_vgg16.PERFORMANCE_PROGRESS_FILE, performance)

    return result['meanIU'] > max_acc



class SemanticSegmentationTrainer(util.Trainer):

    def __init__(self):
        super(SemanticSegmentationTrainer, self).__init__()

    def load_model(self):
        self.saver, self.epoch_step = fcn8s_vgg16.load(self.session)


        for layer in fcn8s_vgg16.layer_list:
            with tf.name_scope(layer):
                tf.summary.histogram('weights', self.session.graph.get_tensor_by_name(layer + '/weights:0'))
                # tf.summary.histogram('biases', self.session.graph.get_tensor_by_name(layer + '/biases:0'))


        self.var_logger = tf.summary.merge_all()

        self.input = self.session.graph.get_tensor_by_name('input:0')
        self.train_phase = self.session.graph.get_tensor_by_name('train_phase:0')
        self.output = self.session.graph.get_tensor_by_name('upscore8_1:0')
        self._step = self.session.graph.get_tensor_by_name('global_step:0')
        loss = self.session.graph.get_tensor_by_name('loss:0')
        # sess.run(tf.global_variables_initializer())

        self.ground_truth = self.session.graph.get_tensor_by_name('ground_truth:0')
        self.train_op = self.session.graph.get_tensor_by_name('train_op:0')
        self.momentum = trainer.session.graph.get_tensor_by_name("momentum:0")
        self.metrics = util.MetricsCalculator(self.session, labels=self.ground_truth, predicts=tf.argmax(self.output, 3), loss=loss, num_classes=dataset.NUM_CLASS, name="Metrics")
        self.session.run(tf.local_variables_initializer())
        # self.coord = tf.train.Coordinator()
        # self.__threads = tf.train.start_queue_runners(coord=self.coord, sess=self.session)


    def close(self):
        # self.coord.request_stop()
        # self.coord.join(self.__threads)
        self.session.close()

    # def train(self):
    #     for i in range(100):
    #         images, _ = data.batch(self.batch_size)
    #         image = images.x
    #         label = images.y
    #
    #         cost, _ = self.session.run([self.loss, self.optimizer], feed_dict={self.input: image, self.ground_truth: label})
    #
    #         print("Iter " + str(i + 1) + ", Loss = " + "{:.6f}".format(cost))


# with tf.Session() as sess:
# saver = tf.train.Saver()

if __name__ == "__main__":


    trainer = SemanticSegmentationTrainer()
    trainer.load_model()

    momentum = min(START_MOMENTUM + (trainer.epoch_step // MOMENTUM_UP_EPOCH_STEP) * MOMENTUM_INCREASE, MAX_MOMENTUM)


    train_data = dataset.PascalVOCSegmentationDataSet(batch_size=BATCH_SIZE, index_path=dataset.DATA_PATH + '/train.txt')
    train_data.load()

    val_data = dataset.PascalVOCSegmentationDataSet(batch_size=BATCH_SIZE,index_path=dataset.DATA_PATH + '/val.txt')
    val_data.load()

    logger = util.TensorflowLogger()

    augmentation_methods = [
        dataset.AUGMENTATION_METHODS['random_flip_horizontal'],
        dataset.AUGMENTATION_METHODS['random_flip_vertical'],
        dataset.AUGMENTATION_METHODS['random_crop']
    ]

    predict = tf.squeeze(tf.argmax(trainer.output, 3))

    saver = tf.train.Saver()
    for i in range(EPOCH_NUM):
        print ("\n==========================================================")
        print ("Trained step : " + str(trainer.step) + ", training progress...")
        result = trainer.train(train_data, augmentation_methods, momentum)

        print ("--------------------------------------------------")
        print ("Epoch : " + str(trainer.epoch_step))
        print ("Loss = " + "{:.6f}".format(result['loss']))
        print ("Pixel accuracy = " + "{:.6f}".format(result['pixel_accuracy']))
        print ("Mean accuracy = " + "{:.6f}".format(result['mean_accuracy']))
        print ("Mean IU = " + "{:.6f}".format(result['meanIU']))
        print ("Train time = " + str(result['train_time']) + " s")


        print ("--------------------------------------------------")
        print ("Validation progress...")
        val_result = trainer.validation(val_data)
        print ("--------------------------------------------------")
        print ("Validation loss = " + "{:.6f}".format(val_result['loss']))
        print ("Validation pixel accuracy = " + "{:.6f}".format(val_result['pixel_accuracy']))
        print ("Validation mean accuracy = " + "{:.6f}".format(val_result['mean_accuracy']))
        print ("Validation mean IU = " + "{:.6f}".format(val_result['meanIU']))

        print ("Saving progress. Please do not terminate the program...")

        logger.log_scalar(result['loss'], trainer.epoch_step, fcn8s_vgg16.PATH + fcn8s_vgg16.MODEL_NAME + '/trainlog', trainer.session.graph, "loss")
        logger.log_scalar(result['pixel_accuracy'], trainer.epoch_step, fcn8s_vgg16.PATH + fcn8s_vgg16.MODEL_NAME + '/trainlog', trainer.session.graph, 'pixel_accuracy')
        logger.log_scalar(result['mean_accuracy'], trainer.epoch_step, fcn8s_vgg16.PATH + fcn8s_vgg16.MODEL_NAME + '/trainlog', trainer.session.graph, 'mean_accuracy')
        logger.log_scalar(result['meanIU'], trainer.epoch_step, fcn8s_vgg16.PATH + fcn8s_vgg16.MODEL_NAME + '/trainlog', trainer.session.graph, 'meanIU')


        logger.log_scalar(val_result['loss'], trainer.epoch_step, fcn8s_vgg16.PATH + fcn8s_vgg16.MODEL_NAME + '/validationlog', trainer.session.graph, "loss")
        logger.log_scalar(val_result['pixel_accuracy'], trainer.epoch_step, fcn8s_vgg16.PATH + fcn8s_vgg16.MODEL_NAME + '/validationlog', trainer.session.graph, 'pixel_accuracy')
        logger.log_scalar(val_result['mean_accuracy'], trainer.epoch_step, fcn8s_vgg16.PATH + fcn8s_vgg16.MODEL_NAME + '/validationlog', trainer.session.graph, 'mean_accuracy')
        logger.log_scalar(val_result['meanIU'], trainer.epoch_step, fcn8s_vgg16.PATH + fcn8s_vgg16.MODEL_NAME + '/validationlog', trainer.session.graph, 'meanIU')
        logger.log_sumary(val_result['variable_log'], trainer.epoch_step, fcn8s_vgg16.PATH + fcn8s_vgg16.MODEL_NAME + '/validationlog', trainer.session.graph)



        better = better_performance(val_result)
        saver.save(trainer.session, fcn8s_vgg16.TRAINING_MODEL_PATH + fcn8s_vgg16.MODEL_NAME, write_meta_graph=True)
        if better == True:
            saver.save(trainer.session, fcn8s_vgg16.BEST_MODEL_PATH + fcn8s_vgg16.MODEL_NAME, write_meta_graph=True)

        if trainer.epoch_step % MOMENTUM_UP_EPOCH_STEP == 0:
            momentum = min(momentum + MOMENTUM_INCREASE, MAX_MOMENTUM)


    trainer.close()
    print("\n\nTrain finished")
