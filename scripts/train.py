from __future__ import print_function
import core.utility as util
import fcn8
import scipy.io
import tensorflow as tf
import dataset
import os
import numpy as np


BATCH_SIZE = 5
STEP_TO_VALIDATION = 100


def better_performance(new_accuracy, new_loss):
    assert os.path.exists(fcn8.PERFORMANCE_PROGRESS_FILE)
    assert new_accuracy >= 0
    assert new_accuracy <= 1
    assert new_loss >= 0

    performance = scipy.io.loadmat(fcn8.PERFORMANCE_PROGRESS_FILE)
    max_acc = -1

    if np.size(performance['accuracy']) > 0:
        max_acc = np.max(performance['accuracy'])

    performance['accuracy'] = np.append(performance['accuracy'], new_accuracy)
    performance['loss'] = np.append(performance['loss'], new_loss)
    scipy.io.savemat(fcn8.PERFORMANCE_PROGRESS_FILE, performance)
    return new_accuracy > max_acc



class ImageSegmentationTrainer(util.Trainer):

    def __init__(self, learning_rate):
        super(ImageSegmentationTrainer, self).__init__(learning_rate)

    def load_model(self):
        self.saver = fcn8.load(self.session)

        for layer in fcn8.layer_list:
            with tf.name_scope(layer):
                tf.summary.histogram('weight', self.session.graph.get_tensor_by_name(layer + '/w:0'))
                tf.summary.histogram('bias', self.session.graph.get_tensor_by_name(layer + '/b:0'))

        self.var_logger = tf.summary.merge_all()

        self.input = self.session.graph.get_tensor_by_name('input:0')
        self.is_training = self.session.graph.get_tensor_by_name('is_training:0')
        self.predict = self.session.graph.get_tensor_by_name('deconv_3:0')
        self._step = self.session.graph.get_tensor_by_name('global_step:0')
        # sess.run(tf.global_variables_initializer())

        self.ground_truth = tf.placeholder(tf.int64, shape=[None, None, None], name='truth')

        self.loss = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.ground_truth, logits=self.predict, name="Loss"))
        corrects = tf.equal(tf.argmax(self.predict, 3), self.ground_truth)
        self.accuracy = tf.reduce_mean(tf.cast(corrects, tf.float32))

        self.train_op = tf.train.MomentumOptimizer(learning_rate=self.learning_rate, momentum=tf.constant(0.9),
                                               use_nesterov=True).minimize(self.loss, self._step)

        momentum_initializers = [var.initializer for var in tf.global_variables() if 'Momentum' in var.name]

        self.session.run(momentum_initializers)


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
    train_data = dataset.PascalVOCSegmentationDataSet(dataset.DATA_PATH + '/train.txt')
    train_data.load()
    train_data.shuffle()

    val_data = dataset.PascalVOCSegmentationDataSet(dataset.DATA_PATH + '/val.txt')
    val_data.load()
    val_data.shuffle()

    trainer = ImageSegmentationTrainer(learning_rate=1e-3)
    trainer.load_model()

    logger = util.TensorflowLogger()

    augmentation_methods = [
        dataset.AUGMENTATION_METHODS['random_flip_horizontal'],
        dataset.AUGMENTATION_METHODS['random_flip_vertical'],
        dataset.AUGMENTATION_METHODS['random_crop']
    ]

    while True:
        print ("\n==========================================================")
        print ("Trained step : " + str(trainer.step) + ", training progress...")
        result = trainer.train(train_data, BATCH_SIZE, STEP_TO_VALIDATION, augmentation_methods)
        print ("--------------------------------------------------")
        print ("Iterations :" + str(STEP_TO_VALIDATION))
        print ("Loss = " + "{:.6f}".format(result['loss']))
        print ("Accuracy = " + "{:.6f}".format(result['accuracy']))

        print ("--------------------------------------------------")
        print ("Validation processing...")
        val_result = trainer.validation(val_data, batch_size=BATCH_SIZE)
        print ("--------------------------------------------------")
        print ("Validation loss = " + "{:.6f}".format(val_result['loss']))
        print ("Validation accuracy = " + "{:.6f}".format(val_result['accuracy']))

        print ("Saving progress. Please do not terminate the program...")

        logger.log_scalar(result['loss'], trainer.step, fcn8.PATH + fcn8.MODEL_NAME + '/trainlog', trainer.session.graph, "loss")
        logger.log_scalar(result['accuracy'], trainer.step, fcn8.PATH + fcn8.MODEL_NAME + '/trainlog', trainer.session.graph, "accuracy")

        logger.log_scalar(val_result['loss'], trainer.step, fcn8.PATH + fcn8.MODEL_NAME + '/validationlog', trainer.session.graph, "loss")
        logger.log_scalar(val_result['accuracy'], trainer.step, fcn8.PATH + fcn8.MODEL_NAME + '/validationlog',trainer.session.graph, "accuracy")
        logger.log_sumary(val_result['variable_log'], trainer.step, fcn8.PATH + fcn8.MODEL_NAME + '/validationlog', trainer.session.graph)


        better = better_performance(val_result['accuracy'], val_result['loss'])
        trainer.saver.save(trainer.session, fcn8.TRAINING_MODEL_PATH + fcn8.MODEL_NAME, write_meta_graph=False)
        if better == True:
            trainer.saver.save(trainer.session, fcn8.BEST_MODEL_PATH + fcn8.MODEL_NAME, write_meta_graph=False)