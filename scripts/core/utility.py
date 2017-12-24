from __future__ import  division
from abc import ABCMeta, abstractmethod
import tensorflow as tf
import datamanager
import datetime


class MetricsCalculator:


    def __init__(self, sess, labels, predicts, loss, num_classes, name):
        assert sess._closed == False
        self.__session = sess
        self.name = name
        self.__var_init = [i.initializer for i in tf.local_variables() if i.name.split('/')[0] == self.name]

        self.accuracy, self.accuracy_op = tf.metrics.accuracy(labels=labels, predictions=predicts)
        self.mean_accuracy, self.mean_accuracy_op = tf.metrics.mean_per_class_accuracy(labels=labels, predictions=predicts, num_classes=num_classes)
        self.meanIU, self.meanIU_op = tf.metrics.mean_iou(labels=labels, predictions=predicts, num_classes=num_classes)
        self.loss, self.loss_op = tf.metrics.mean(values=loss)


    def reset(self):
        self.__session.run(self.__var_init)





class Trainer:

    __metaclass__ = ABCMeta

    def __init__(self):
        self.saver = None
        self.train_phase = None
        self.train_op = None
        self.predict = None
        self.input = None
        self.ground_truth = None
        self._step = None
        self.var_logger = None
        self.epoch_step = None
        self.metrics = None
        self.momentum = None

        config = tf.ConfigProto(
            device_count={'GPU': 1, 'CPU': 1},
            allow_soft_placement=True,
            log_device_placement = True
        )
        self.session = tf.Session(config=config)



    @abstractmethod
    def load_model(self):
        pass

    @property
    def step(self):
        assert self.session != None
        assert self._step != None
        return tf.train.global_step(self.session, self._step)


    def validation(self, dataset):
        assert self.session != None
        assert self._step != None
        assert dataset.size > 0


        dataset.reset()
        self.metrics.reset()
        while True:
            batch = dataset.batch()
            if batch is not None:
                images = batch.x
                truths = batch.y
                self.session.run([self.metrics.loss_op, self.metrics.accuracy_op, self.metrics.mean_accuracy_op, self.metrics.meanIU_op],
                                 feed_dict={self.input: images, self.ground_truth: truths, self.train_phase: False})
            else:
                break

        var_log = self.session.run(self.var_logger)
        loss = self.session.run(self.metrics.loss)
        pixel_accuracy = self.session.run(self.metrics.accuracy)
        mean_accuracy = self.session.run(self.metrics.mean_accuracy)
        meanIU = self.session.run(self.metrics.meanIU)

        return {'loss': loss, 'pixel_accuracy': pixel_accuracy, 'mean_accuracy': mean_accuracy, 'meanIU': meanIU, 'variable_log': var_log}



    def train(self, dataset, augmentation_methods, momentum = 0.9):

        assert self.session != None
        assert self._step != None
        assert dataset.size > 0


        dataset.reset()
        dataset.shuffle()
        self.metrics.reset()
        start = datetime.datetime.now()
        while True:
            batch = dataset.batch()
            if batch is not None:
                batch = datamanager.data_augment(batch, augmentation_methods)
                images = batch.x
                truths = batch.y

                self.session.run([self.train_op, self.metrics.loss_op, self.metrics.accuracy_op, self.metrics.mean_accuracy_op, self.metrics.meanIU_op], feed_dict={self.input: images, self.ground_truth: truths, self.train_phase: True, self.momentum : momentum})

            else:
                break

        finish = datetime.datetime.now()
        train_time = (finish - start).total_seconds()
        self.epoch_step += 1

        loss = self.session.run(self.metrics.loss)
        pixel_accuracy = self.session.run(self.metrics.accuracy)
        mean_accuracy = self.session.run(self.metrics.mean_accuracy)
        meanIU = self.session.run(self.metrics.meanIU)

        return {'loss': loss, 'pixel_accuracy': pixel_accuracy, 'mean_accuracy': mean_accuracy, 'meanIU': meanIU, 'train_time': train_time}



class TensorflowLogger:

    def __init__(self):
        self.__loggers = dict()


    def log_scalar(self, value, step, log_path, graph, name):
        if log_path not in self.__loggers:
            self.__loggers[log_path] = tf.summary.FileWriter(log_path, graph)

        assert isinstance(self.__loggers[log_path], tf.summary.FileWriter)

        sumary = tf.Summary(value=[tf.Summary.Value(tag=name, simple_value= value)])
        self.__loggers[log_path].add_summary(sumary, step)


    def log_sumary(self, summary, step, log_path, graph):
        if log_path not in self.__loggers:
            self.__loggers[log_path] = tf.summary.FileWriter(log_path, graph)

        assert isinstance(self.__loggers[log_path], tf.summary.FileWriter)
        self.__loggers[log_path].add_summary(summary, step)


