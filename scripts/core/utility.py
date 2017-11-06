from __future__ import  division
from abc import ABCMeta, abstractmethod
import tensorflow as tf
import datamanager




class Trainer:

    __metaclass__ = ABCMeta

    def __init__(self, learning_rate):
        self.learning_rate = learning_rate
        self.saver = None
        self.is_training = None
        self.train_op = None
        self.loss = None
        self.accuracy = None
        self.predict = None
        self.input = None
        self.ground_truth = None
        self._step = None
        self.var_logger = None
        self.session = tf.Session()
        assert self.session._closed == False


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

        # Set up statistic values
        val_loss = 0
        val_accuracy = 0
        val_iters = 0

        while True:
            batch, end = dataset.batch(self.session)
            images = batch.x
            truths = batch.y
            loss, accuracy = self.session.run([self.loss, self.accuracy],
                            feed_dict={self.input: images, self.ground_truth: truths, self.is_training: False})
            val_loss += loss
            val_accuracy += accuracy
            val_iters += 1
            print val_iters

            if end == True:
                break

        var_log = self.session.run(self.var_logger)

        val_loss = val_loss / val_iters
        val_accuracy = val_accuracy / val_iters

        return {'loss': val_loss, 'accuracy': val_accuracy, 'variable_log': var_log}



    def train(self, dataset, iters, augmentation_methods):

        assert self.session != None
        assert self._step != None
        assert dataset.size > 0

        train_loss = 0
        train_accuracy = 0
        train_iters = 0


        for i in range(iters):
            batch, end = dataset.batch(self.session)
            batch = datamanager.data_augment(batch, augmentation_methods)
            images = batch.x
            truths = batch.y
            _, loss, accuracy = self.session.run([self.train_op, self.loss, self.accuracy], feed_dict={self.input: images, self.ground_truth: truths, self.is_training: True})
            train_loss += loss
            train_accuracy += accuracy
            train_iters += 1
            if end == True:
                dataset.shuffle()


        train_loss = train_loss / train_iters
        train_accuracy = train_accuracy / train_iters

        return {'loss': train_loss, 'accuracy': train_accuracy}



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


