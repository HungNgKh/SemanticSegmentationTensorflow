import core.neuralnetwork as nn
import os, sys, shutil
import tensorflow as tf
import tensorflow.contrib.layers as layer
import dataset
import argparse
import scipy.io


PATH = "/home/khanhhung/deeplearning/saved/SEG/"
MODEL_NAME = "fcn8"
TRAINING_MODEL_PATH = PATH + MODEL_NAME + '/training/'
BEST_MODEL_PATH = PATH + MODEL_NAME + '/best/'
PERFORMANCE_PROGRESS_FILE = PATH + MODEL_NAME + '/performance_progress.mat'




layer_list = [
    "conv1_1", "conv1_2",
    "conv2_1", "conv2_2",
    "conv3_1", "conv3_2",
    "conv4_1", "conv4_2",
    "conv5_1", "conv5_2",
    "conv6",
    "conv7",
    "deconv1","pool4_conv",
    "deconv2","pool3_conv",
    "deconv3"
]




def __build():

    if(os.path.exists(TRAINING_MODEL_PATH + MODEL_NAME + '.meta')):
        print "ERROR! Tried to build an existed model"
        sys.exit()

    if not os.path.exists(TRAINING_MODEL_PATH):
        os.makedirs(TRAINING_MODEL_PATH)

    if not os.path.exists(BEST_MODEL_PATH):
        os.makedirs(BEST_MODEL_PATH)


    # Define variables of graph
    variables_dict = {}



    with tf.variable_scope('conv1_1'):
        variables_dict['conv1_1_w'] = tf.get_variable(name='w', shape=[3, 3, 3, 32],
                                              initializer=layer.xavier_initializer())
        variables_dict['conv1_1_b'] = tf.get_variable(name='b', shape=[32], initializer=tf.zeros_initializer)

    with tf.variable_scope('conv1_2'):
        variables_dict['conv1_2_w'] = tf.get_variable(name='w', shape=[3, 3, 32, 32],
                                              initializer=layer.xavier_initializer())
        variables_dict['conv1_2_b'] = tf.get_variable(name='b', shape=[32], initializer=tf.zeros_initializer)



    with tf.variable_scope('conv2_1'):
        variables_dict['conv2_1_w'] = tf.get_variable(name='w', shape=[3, 3, 32, 64],
                                              initializer=layer.xavier_initializer())
        variables_dict['conv2_1_b'] = tf.get_variable(name='b', shape=[64], initializer=tf.zeros_initializer)

    with tf.variable_scope('conv2_2'):
        variables_dict['conv2_2_w'] = tf.get_variable(name='w', shape=[3, 3, 64, 64],
                                              initializer=layer.xavier_initializer())
        variables_dict['conv2_2_b'] = tf.get_variable(name='b', shape=[64], initializer=tf.zeros_initializer)



    with tf.variable_scope('conv3_1'):
        variables_dict['conv3_1_w'] = tf.get_variable(name='w', shape=[3, 3, 64, 128],
                                              initializer=layer.xavier_initializer())
        variables_dict['conv3_1_b'] = tf.get_variable(name='b', shape=[128], initializer=tf.zeros_initializer)

    with tf.variable_scope('conv3_2'):
        variables_dict['conv3_2_w'] = tf.get_variable(name='w', shape=[3, 3, 128, 128],
                                              initializer=layer.xavier_initializer())
        variables_dict['conv3_2_b'] = tf.get_variable(name='b', shape=[128], initializer=tf.zeros_initializer)



    with tf.variable_scope('conv4_1'):
        variables_dict['conv4_1_w'] = tf.get_variable(name='w', shape=[3, 3, 128, 256],
                                              initializer=layer.xavier_initializer())
        variables_dict['conv4_1_b'] = tf.get_variable(name='b', shape=[256], initializer=tf.zeros_initializer)

    with tf.variable_scope('conv4_2'):
        variables_dict['conv4_2_w'] = tf.get_variable(name='w', shape=[3, 3, 256, 256],
                                              initializer=layer.xavier_initializer())
        variables_dict['conv4_2_b'] = tf.get_variable(name='b', shape=[256], initializer=tf.zeros_initializer)



    with tf.variable_scope('conv5_1'):
        variables_dict['conv5_1_w'] = tf.get_variable(name='w', shape=[3, 3, 256, 256],
                                              initializer=layer.xavier_initializer())
        variables_dict['conv5_1_b'] = tf.get_variable(name='b', shape=[256], initializer=tf.zeros_initializer)

    with tf.variable_scope('conv5_2'):
        variables_dict['conv5_2_w'] = tf.get_variable(name='w', shape=[3, 3, 256, 256],
                                              initializer=layer.xavier_initializer())
        variables_dict['conv5_2_b'] = tf.get_variable(name='b', shape=[256], initializer=tf.zeros_initializer)



    with tf.variable_scope('conv6'):
        variables_dict['conv6_w'] = tf.get_variable(name='w', shape=[7, 7, 256, 2048],
                                              initializer=layer.xavier_initializer())
        variables_dict['conv6_b'] = tf.get_variable(name='b', shape=[2048], initializer=tf.zeros_initializer)



    with tf.variable_scope('conv7'):
        variables_dict['conv7_w'] = tf.get_variable(name='w', shape=[1, 1, 2048, 2048],
                                              initializer=layer.xavier_initializer())
        variables_dict['conv7_b'] = tf.get_variable(name='b', shape=[2048], initializer=tf.zeros_initializer)



    with tf.variable_scope('conv8'):
        variables_dict['conv8_w'] = tf.get_variable(name='w', shape=[1, 1, 2048, dataset.NUM_CLASS],
                                              initializer=layer.xavier_initializer())
        variables_dict['conv8_b'] = tf.get_variable(name='b', shape=[dataset.NUM_CLASS], initializer=tf.zeros_initializer)



    with tf.variable_scope('deconv1'):
        variables_dict['deconv1_w'] = tf.get_variable(name='w', shape=[4, 4, dataset.NUM_CLASS, dataset.NUM_CLASS],
                                              initializer=layer.xavier_initializer())
        variables_dict['deconv1_b'] = tf.get_variable(name='b', shape=[dataset.NUM_CLASS], initializer=tf.zeros_initializer)



    with tf.variable_scope('pool4_conv'):
        variables_dict['pool4_conv_w'] = tf.get_variable(name='w', shape=[1, 1, 256, dataset.NUM_CLASS],
                                              initializer=layer.xavier_initializer())
        variables_dict['pool4_conv_b'] = tf.get_variable(name='b', shape=[dataset.NUM_CLASS], initializer=tf.zeros_initializer)



    with tf.variable_scope('deconv2'):
        variables_dict['deconv2_w'] = tf.get_variable(name='w', shape=[4, 4, dataset.NUM_CLASS, dataset.NUM_CLASS],
                                              initializer=layer.xavier_initializer())
        variables_dict['deconv2_b'] = tf.get_variable(name='b', shape=[dataset.NUM_CLASS], initializer=tf.zeros_initializer)



    with tf.variable_scope('pool3_conv'):
        variables_dict['pool3_conv_w'] = tf.get_variable(name='w', shape=[1, 1, 128, dataset.NUM_CLASS],
                                              initializer=layer.xavier_initializer())
        variables_dict['pool3_conv_b'] = tf.get_variable(name='b', shape=[dataset.NUM_CLASS], initializer=tf.zeros_initializer)



    with tf.variable_scope('deconv3'):
        variables_dict['deconv3_w'] = tf.get_variable(name='w', shape=[16, 16, dataset.NUM_CLASS, dataset.NUM_CLASS],
                                              initializer=layer.xavier_initializer())
        variables_dict['deconv3_b'] = tf.get_variable(name='b', shape=[dataset.NUM_CLASS], initializer=tf.zeros_initializer)






    #Define operation flow of graph

    input = tf.placeholder(tf.float32, shape=[None, None, None, 3], name="input")
    is_training = tf.placeholder(tf.bool, name="is_training")


    conv1_1 = nn.conv2d("conv1_1", input, variables_dict['conv1_1_w'], 1, variables_dict['conv1_1_b'])
    relu1_1 = tf.nn.relu(conv1_1, "relu1_1")
    conv1_2 = nn.conv2d("conv1_2", relu1_1, variables_dict['conv1_2_w'], 1, variables_dict['conv1_2_b'])
    relu1_2 = tf.nn.relu(conv1_2, "relu1_2")
    pool1 = nn.max_pool("pool1", relu1_2, 2)



    conv2_1 = nn.conv2d("conv2_1", pool1, variables_dict['conv2_1_w'], 1, variables_dict['conv2_1_b'])
    relu2_1 = tf.nn.relu(conv2_1, "relu2_1")
    conv2_2 = nn.conv2d("conv2_2", relu2_1, variables_dict['conv2_2_w'], 1, variables_dict['conv2_2_b'])
    relu2_2 = tf.nn.relu(conv2_2, "relu2_2")
    pool2 = nn.max_pool("pool2", relu2_2, 2)



    conv3_1 = nn.conv2d("conv3_1", pool2, variables_dict['conv3_1_w'], 1, variables_dict['conv3_1_b'])
    relu3_1 = tf.nn.relu(conv3_1, "relu3_1")
    conv3_2 = nn.conv2d("conv3_2", relu3_1, variables_dict['conv3_2_w'], 1, variables_dict['conv3_2_b'])
    relu3_2 = tf.nn.relu(conv3_2, "relu3_2")
    pool3 = nn.max_pool("pool3", relu3_2, 2)



    conv4_1 = nn.conv2d("conv4_1", pool3, variables_dict['conv4_1_w'], 1, variables_dict['conv4_1_b'])
    relu4_1 = tf.nn.relu(conv4_1, "relu4_1")
    conv4_2 = nn.conv2d("conv4_2", relu4_1, variables_dict['conv4_2_w'], 1, variables_dict['conv4_2_b'])
    relu4_2 = tf.nn.relu(conv4_2, "relu4_2")
    pool4 = nn.max_pool("pool4", relu4_2, 2)



    conv5_1 = nn.conv2d("conv5_1", pool4, variables_dict['conv5_1_w'], 1, variables_dict['conv5_1_b'])
    relu5_1 = tf.nn.relu(conv5_1, "relu5_1")
    conv5_2 = nn.conv2d("conv5_2", relu5_1, variables_dict['conv5_2_w'], 1, variables_dict['conv5_2_b'])
    relu5_2 = tf.nn.relu(conv5_2, "relu5_2")
    pool5 = nn.max_pool("pool5", relu5_2, 2)



    conv6 = nn.conv2d("conv6", pool5, variables_dict['conv6_w'], 1, variables_dict['conv6_b'])
    relu6 = tf.nn.relu(conv6, "relu6")

    conv7 = nn.conv2d("conv7", relu6, variables_dict['conv7_w'], 1, variables_dict['conv7_b'])
    relu7 = tf.nn.relu(conv7, "relu7")

    conv8 = nn.conv2d("conv8", relu7, variables_dict['conv8_w'], 1, variables_dict['conv8_b'])
    relu8 = tf.nn.relu(conv8, "relu8")



    pool4_conv = nn.conv2d("pool4_conv", pool4, variables_dict['pool4_conv_w'], 1, variables_dict['pool4_conv_b'])
    deconv_1 = nn.transpose_conv2d("deconv_1", relu8, variables_dict['deconv1_w'], 2, variables_dict['deconv1_b'], tf.shape(pool4_conv))
    fuse_1 = deconv_1 + pool4_conv



    pool3_conv = nn.conv2d("pool3_conv", pool3, variables_dict['pool3_conv_w'], 1, variables_dict['pool3_conv_b'])
    deconv_2 = nn.transpose_conv2d("deconv_2", fuse_1, variables_dict['deconv2_w'], 2, variables_dict['deconv2_b'], tf.shape(pool3_conv))
    fuse_2 = deconv_2 + pool3_conv



    shape_adapt = tf.constant([0, 0, 0, dataset.NUM_CLASS - 3])
    output_shape = tf.shape(input) + shape_adapt

    deconv_3 = nn.transpose_conv2d("deconv_3", fuse_2, variables_dict['deconv3_w'], 8, variables_dict['deconv3_b'],
                                   output_shape)

    global_step = tf.get_variable("global_step", shape=[] ,dtype=tf.int32, initializer=tf.zeros_initializer, trainable=False)

    # print tf.global_variables()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        saver.save(sess, TRAINING_MODEL_PATH + MODEL_NAME)
        saver.save(sess, BEST_MODEL_PATH + MODEL_NAME)

    if os.path.exists(TRAINING_MODEL_PATH + MODEL_NAME + '.meta'):
        print "Successfully create fcn8 neural network model"
    else:
        print "Failed to create fcn8 neural network model"
        return

    performances = {'loss': [], 'accuracy': []}
    scipy.io.savemat(PERFORMANCE_PROGRESS_FILE, performances)



def load(sess):

    if os.path.exists(TRAINING_MODEL_PATH + MODEL_NAME + '.meta'):
        new_saver = tf.train.new_saver = tf.train.import_meta_graph(TRAINING_MODEL_PATH  + MODEL_NAME + '.meta')
        new_saver.restore(sess, tf.train.latest_checkpoint(TRAINING_MODEL_PATH))
        return new_saver

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

    parser = argparse.ArgumentParser(description="Setup fcn8 neural network")
    parser.add_argument('command', choices=function_map.keys())

    args = parser.parse_args()
    func = function_map[args.command]
    func()


if __name__ == "__main__":
    __main()

    # build()