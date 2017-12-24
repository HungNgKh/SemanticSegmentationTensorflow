import tensorflow as tf
import scripts.fcn8s_vgg16 as fcn8s
import tensorflow.contrib.layers as layer
import scipy.io
import numpy as np
import os


# def trained_epoch_num():
#     if os.path.exists(fcn8s.PERFORMANCE_PROGRESS_FILE):
performance = scipy.io.loadmat(fcn8s.PERFORMANCE_PROGRESS_FILE)
print performance
#         return np.size(performance['accuracy'])
#     else:


