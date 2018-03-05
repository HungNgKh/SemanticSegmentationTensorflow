import core.neuralnetwork as nn
import os, sys, shutil
import tensorflow as tf
import tensorflow.contrib.layers as layer
import dataset
from pretrained_model import vgg16
import argparse, os
import scipy.io
import numpy as np


PATH = os.path.dirname(os.path.abspath(__file__)) + "/../data/progress/"
MODEL_NAME = "fcn8s_vgg16"
TRAINING_MODEL_PATH = PATH + MODEL_NAME + '/training/'
BEST_MODEL_PATH = PATH + MODEL_NAME + '/best/'
PERFORMANCE_PROGRESS_FILE = PATH + MODEL_NAME + '/performance_progress.mat'

LEARNING_RATE = 5e-4
DECAY_RATE = 0.95
DECAY_STEP = 2000
WEIGHT_DECAY = 5e-4

layer_list = [
    "conv6",
    "conv7",
    "score_fr",
    "upscore_fr","score_pool4",
    "upscore_pool4","score_pool3",
    "upscore8"
]




def __build():

    if(os.path.exists(TRAINING_MODEL_PATH + MODEL_NAME + '.meta')):
        print "ERROR! Tried to build an existed model"
        sys.exit()

    if not os.path.exists(TRAINING_MODEL_PATH):
        os.makedirs(TRAINING_MODEL_PATH)

    if not os.path.exists(BEST_MODEL_PATH):
        os.makedirs(BEST_MODEL_PATH)



    images = tf.placeholder(tf.float32, shape=[None, None, None, 3], name="input")
    train_phase = tf.placeholder(tf.bool, name='train_phase')

    # with tf.device("/cpu:0"):
    pretrained_vgg16 = vgg16.vgg16(images)

    pool5 = pretrained_vgg16.pool5

    conv6 = nn.conv2d('conv6', pool5, [7, 7, 512, 4096], 1, tf.truncated_normal_initializer(stddev=1e-3), False, False)
    relu6 = tf.nn.relu(conv6, name='relu6')
    drop6 = nn.dropout(relu6, train_phase, 0.6)

    conv7 = nn.conv2d('conv7', drop6, [1, 1, 4096, 4096], 1, tf.truncated_normal_initializer(stddev=1e-3), False, False)
    relu7 = tf.nn.relu(conv7, name='relu7')
    drop7 = nn.dropout(relu7, train_phase, 0.6)

    score_fr = nn.conv2d('score_fr', drop7, [1, 1, 4096, dataset.NUM_CLASS], 1, tf.truncated_normal_initializer(stddev=1e-3), True, True)
    relu_score_fr = tf.nn.relu(score_fr, name='relu_score_fr')

    score_pool4 = nn.conv2d('score_pool4', pretrained_vgg16.pool4, [1, 1, 512, dataset.NUM_CLASS], 1, tf.zeros_initializer, True, True)
    upscore_fr = nn.transpose_conv2d('upscore_fr', relu_score_fr, [4, 4, dataset.NUM_CLASS, dataset.NUM_CLASS], 2, tf.shape(score_pool4), True, True)
    fuse_pool4 = tf.add(score_pool4, upscore_fr, name='fuse_pool4')

    score_pool3 = nn.conv2d('score_pool3', pretrained_vgg16.pool3, [1, 1, 256, dataset.NUM_CLASS], 1,
                            tf.zeros_initializer, True, True)
    upscore_pool4 = nn.transpose_conv2d('upscore_pool4', fuse_pool4, [4, 4, dataset.NUM_CLASS, dataset.NUM_CLASS], 2, tf.shape(score_pool3), True, True)
    fuse_pool3 = tf.add(score_pool3, upscore_pool4, name='fuse_pool3')

    ouput_shape = tf.shape(images) + tf.constant([0, 0, 0, dataset.NUM_CLASS - images.get_shape().as_list()[-1]])
    upscore8 = nn.transpose_conv2d('upscore8', fuse_pool3, [16, 16, dataset.NUM_CLASS, dataset.NUM_CLASS], 8, ouput_shape, True, True)


    ground_truth = tf.placeholder(tf.int32, [None, None, None], name='ground_truth')

    with tf.device('/GPU:0'):
        weight_norm = tf.reduce_sum(WEIGHT_DECAY * tf.stack(
            [tf.nn.l2_loss(tf.get_default_graph().get_tensor_by_name(layer_name + '/weights:0')) for layer_name in layer_list]),
                                    name="weight_decay_loss")

        loss = tf.add(tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(labels=ground_truth, logits=upscore8)), weight_norm,
            name="loss")

        momentum = tf.placeholder(dtype=tf.float32, name='momentum')

        global_step = tf.get_variable("global_step", shape=[] ,dtype=tf.int32, initializer=tf.zeros_initializer, trainable=False)

        start_learning_rate = LEARNING_RATE
        learning_rate = tf.train.exponential_decay(start_learning_rate, global_step, DECAY_STEP, DECAY_RATE, staircase=True)

    # with tf.device('/CPU:0'):
    train_op = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=momentum, use_nesterov = True).minimize(loss, global_step, name = "train_op", colocate_gradients_with_ops=True)

    config = tf.ConfigProto(
        device_count={'CPU': 1, 'GPU': 1},
        allow_soft_placement=True,
        log_device_placement=True
    )
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        pretrained_vgg16.load_weights(vgg16.WEIGHT_PATH, sess)
        # pretrained_vgg16.check_weight(vgg16.WEIGHT_PATH, sess)
        print sess.run(tf.contrib.memory_stats.BytesInUse())
        saver = tf.train.Saver()
        saver.save(sess, TRAINING_MODEL_PATH + MODEL_NAME)
        saver.save(sess, BEST_MODEL_PATH + MODEL_NAME)

    if os.path.exists(TRAINING_MODEL_PATH + MODEL_NAME + '.meta'):
        print "Successfully create " + MODEL_NAME + " neural network model"
    else:
        print "Failed to create " + MODEL_NAME + " neural network model"
        return

    performances = {'loss': [], 'pixel_accuracy': [], 'mean_accuracy': [], 'meanIU': []}
    scipy.io.savemat(PERFORMANCE_PROGRESS_FILE, performances)



def load(sess):

    def trained_epoch_num():
        if os.path.exists(PERFORMANCE_PROGRESS_FILE):
            performance = scipy.io.loadmat(PERFORMANCE_PROGRESS_FILE)
            return np.size(performance['meanIU'])
        else:
            return 0

    if os.path.exists(TRAINING_MODEL_PATH + MODEL_NAME + '.meta'):
        new_saver = tf.train.import_meta_graph(TRAINING_MODEL_PATH  + MODEL_NAME + '.meta')
        new_saver.restore(sess, tf.train.latest_checkpoint(TRAINING_MODEL_PATH))
        return tf.train.Saver(tf.global_variables()), trained_epoch_num()

    else:
        print "Model not found!"
        sys.exit()



def __clear():

    if os.path.exists(PATH + MODEL_NAME):
        shutil.rmtree(PATH + MODEL_NAME)





def __main():
    def __rebuild():
        if (os.path.exists(TRAINING_MODEL_PATH + MODEL_NAME + '.meta')):
            __clear()

        __build()


    function_map = {
        'build': __build,
        'rebuild': __rebuild,
        'delete' : __clear
    }

    parser = argparse.ArgumentParser(description="Setup " + MODEL_NAME + " neural network")
    parser.add_argument('command', choices=function_map.keys())

    args = parser.parse_args()
    func = function_map[args.command]
    func()


if __name__ == "__main__":
    __main()

    # build()