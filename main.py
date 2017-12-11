import tensorflow as tf
import tensorflow.contrib.rnn as rnn
import os
import numpy as np
import time
import sys
from utils._convert_Youtube8M_tfrecord_to_numpy import _convert_Youtube8M_tfrecord_to_numpy
from models import two_stream_lstm_model



# !Check if tensorflow version == '1.4.0', API
assert tf.__version__ == '1.4.0'



# !Set system default encoding method == 'utf-8'
reload(sys)
sys.setdefaultencoding('utf-8')




# assign computation tasks excuted by GPU:'0',
# uncomment if you want to use all resources i.e. multi-GPUs and 100% memory-on-board on your server
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"




# optional:
    # with tf.device('/gpu:0'):
    #   tensor/operations definition...
    # with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
    #   sess.run(op)






# !Instantialize a two_stream_lstm_model object...
with tf.Session() as sess:
    two_steam_lstm_net = two_stream_lstm_model.two_stream_lstm_model(sess=sess)

    time_start = time.clock()

    two_steam_lstm_net._train_phase()

    time_end = time.clock()
    print("!Module Runtime is %.3f seconds..." % (time_end-time_start))
