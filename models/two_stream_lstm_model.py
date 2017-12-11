import tensorflow as tf
import tensorflow.contrib.rnn as rnn
import numpy as np
import sys
import os
from utils import utils

class two_stream_lstm_model(object):

    def __init__(self, sess, train_params={ 'batch_size':   16,
                                            'base_lr_rate': 1e-4 ,
                                            'nframes':      300,
                                            'dim_rgb_feat': 1024,
                                            'dim_audio_feat': 128,
                                            'num_of_class': 4716,
                                            'model_dir':    "./two_stream_models",
                                            'data_dir':     "./data",
                                            'max_iteration': 1000000}):

        # parse the training parameters e.g. batch_size and base_learning_rate
        self.batch_size   = train_params['batch_size']
        self.base_lr_rate = train_params['base_lr_rate']
        self.model_dir    = train_params['model_dir']
        self.data_dir     = train_params['data_dir']

        self.max_iteration = train_params['max_iteration']

        self.nframes      = train_params['nframes']

        self.num_of_class = train_params['num_of_class']

        self.dim_rgb_feat   = train_params['dim_rgb_feat']
        self.dim_audio_feat = train_params['dim_audio_feat']

        # tf.session
        self.sess = sess

        # build-up the two_stream_lstm_model
        self._build_model()








    def _load_youtube8M_tfrecord_data(self):
        '''
            feature map dimension order: [N, W, H, C]
                                               i.e. N- num of mini-batch samples
                                                    W- num of frames
                                                    H- dimension of feature e.g. 'rgb' or 'audio'
                                                    C- num of feature map channels e.g. 1 (omitted)
        '''

        file_list = []

        for file in os.listdir(self.data_dir):
            filename = os.path.join(self.data_dir, file)


            if os.path.isfile(filename):
                file_list.append(filename)
            else:
                continue

        # !Define a filename queue to store the filenames...
        filename_queue = tf.train.string_input_producer(file_list, num_epochs=1, shuffle=False)

        # !Define a tfrecord file reader...
        reader=tf.TFRecordReader()

        # !Read .tfrecord file into serial_example...
        _, serial_example = reader.read(filename_queue)

        # !Parse serial_example into tensor...
        context, features = tf.parse_single_sequence_example(serial_example,
                                                                  context_features={
                                                                      'video_id': tf.VarLenFeature(tf.string),
                                                                      'labels': tf.VarLenFeature(tf.int64)
                                                                  },
                                                                  sequence_features={
                                                                      'rgb': tf.FixedLenSequenceFeature([], tf.string),
                                                                      'audio': tf.FixedLenSequenceFeature([], tf.string)
                                                                  },
                                                                  example_name=" ")


        # !Split the sequence_features and context_features dictionary into 1.rgb_feature, 2.audio_feature, 3.video_id, 4.labels...
        rgb_feat   = features['rgb']
        audio_feat = features['audio']

        video_id = context['video_id']
        labels   = (tf.cast(tf.sparse_to_dense(context["labels"].values, (self.num_of_class, ), 1, validate_indices=False), tf.bool))



        # !Decode examples into tensor...
        decode_rgb   = tf.reshape(tf.cast(tf.decode_raw(bytes=rgb_feat, out_type=tf.uint8), tf.float32), [-1, self.dim_rgb_feat])
        decode_audio = tf.reshape(tf.cast(tf.decode_raw(bytes=audio_feat, out_type=tf.uint8), tf.float32), [-1, self.dim_audio_feat])

        # !Dequantize...
        dequantize_rgb = utils._dequantize(decode_rgb)
        dequantize_audio = utils._dequantize(decode_audio)

        # !Resize, i.e. crop or padding with 0...
        resize_rgb = utils._resize_axis(inp_feat=dequantize_rgb, fill_value=0, new_size=self.nframes, axis=0)
        resize_audio = utils._resize_axis(inp_feat=dequantize_audio, fill_value=0, new_size=self.nframes, axis=0)

        # !Create mini-batch data...
        [rgb_batch, audio_batch, labels_batch, video_id_batch] = tf.train.shuffle_batch([resize_rgb, resize_audio, labels, video_id],
                                                                                        batch_size=self.batch_size,
                                                                                        capacity=self.batch_size*5,
                                                                                        min_after_dequeue=1,
                                                                                        allow_smaller_final_batch=True)

        return rgb_batch, audio_batch, labels_batch, video_id_batch









    def _embedding_layer(self, inp_feat, emb_size, name_scope):
        '''
            Function:
                        _embedding_layer without parameter sharing...
            Input:
                        [1] <float32 tensor> inp_feat: dimension <batch_size, nframes, dim_feat>
                        [2] <int tensor>     emb_size: default 512
                        [3] <string tensor>  name_scope
            Output:
                        <tensor>  emb_feat

        '''

        # get the dimension of input feature tensor...
        dim_feat = inp_feat.shape[2].value



        with tf.name_scope(name_scope):
            emb_weight = tf.Variable(dtype=tf.float32, initial_value=tf.random_normal(shape=[dim_feat, emb_size], stddev=1e-3), name='emb_weight')
            emb_bias   = tf.Variable(dtype=tf.float32, initial_value=tf.zeros(shape=[emb_size]), name='emb_bias')

            inp_feat = tf.reshape(inp_feat, shape=[self.batch_size* self.nframes, dim_feat])

            out_feat = tf.matmul(inp_feat, emb_weight) + emb_bias

            out_feat = tf.reshape(out_feat, shape=[self.batch_size, self.nframes, emb_size])

            return out_feat












    def _bidirectional_LSTM_cell(self, inp_feat, num_of_hidden_units, num_of_cell_state, var_scope):
        '''
            Function:
                        _bidirectional_LSTM_cell
                            i.e. referred from 'https://github.com/jonrein/tensorflow_CTC_example/blob/master/bdlstm_train.py'
            Input:
                        [1] <tensor float32> inp_feat
                                                dimension -> [batch_size, nframes, dim_feat]
                        [2] <int32>          num_of_hidden_units
                        [3] <int32>          num_of_cell_state
                        [4] <string>         var_scope: variable scope
            Output:
                        <tensor float32 list>    out_feat
                                                    dimension -> [batch_size, nframes, dim_feat]
        '''

        # parameters:
        #               [1] batch_size
        #               [2] nframes:   number of time steps
        #               [3] dim_feat:  dimension of feature, 'rgb'=1024, 'audio'=128


        assert num_of_cell_state == inp_feat.shape[1].value


        inp_feat = tf.transpose(inp_feat, perm=[1,0,2])

        inp_feat = tf.unstack(inp_feat,axis=0)

        # define forward LSTM cell...
        forward_lstm_cell = rnn.LSTMCell(num_units      =num_of_hidden_units,
                                         use_peepholes  =True,
                                         state_is_tuple =True)

        # define backward LSTM cell...
        backward_lstm_cell = rnn.LSTMCell(num_units      =num_of_hidden_units,
                                          use_peepholes  =True,
                                          state_is_tuple =True)

        # define bidirectional LSTM network...
        #          turple: [outputs, final_cell_state_forward, final_cell_state_backward]...
        bidi_lstm, cell_state_fw, cell_state_bw  = rnn.static_bidirectional_rnn(cell_fw =forward_lstm_cell,
                                                                                cell_bw =backward_lstm_cell,
                                                                                inputs  =inp_feat,
                                                                                scope   =var_scope,
                                                                                dtype   =tf.float32)

        bidi_lstm = tf.transpose(tf.stack(bidi_lstm), perm=[1,0,2])


        return bidi_lstm









    def _attention_module(self, inp_feat, name_scope):
        '''
            Function:
                    _attention_module
                        i.e. [1] self-attention: inp_feat [nframes, dim_feat] -> self-attention -> frame_weight [nframes, 1]
                             [2] scaling:        inp_feat .* frame_weight
        '''
        dim_feat = inp_feat.shape[2].value

        inp_feat = tf.reshape(inp_feat, shape=[-1, dim_feat])

        with tf.name_scope(name_scope):
            self_att = tf.Variable(dtype=tf.float32, initial_value=tf.random_normal(shape=[dim_feat, 1]), name='self_attention')

        # tf.matmul(x, y): matrix multiplication
        # e.g.  output = tf.matmul(a,b)
        #           i.e. output[...,i,j]=sum_k(a[...,i,k]*b[...,k,j])
        frame_weight = tf.matmul(inp_feat, self_att)

        # tf.multiply(x, y): element-wise multiplication x*y...
        out_feat = tf.multiply(inp_feat, frame_weight)

        out_feat = tf.reshape(out_feat, shape=[self.batch_size, self.nframes, dim_feat])

        return out_feat







    def _feature_aggregation_module(self, inp_feat):
        '''
            Function:
                        _feature_aggregation_module
                            i.e. aggregate a sequence of features along the temporal direction...
            Input:
                        inp_feat, dimension-> [batch_size, nframes, dim_feat]
            Output:
                        out_feat, dimension-> [batch_size, 1, dim_feat]
        '''

        out_feat = tf.reduce_sum(inp_feat, reduction_indices=1)

        return out_feat






    def _fc_layer(self, inp_feat, num_of_outputs, var_scope):
        '''
            Function:
                        _fc_layer
            Input:
                        [1] inp_feat, dimension->[batch_size, dim_feat]
                        [2] num_of_outputs
                        [3] var_scope: reuse=True
            Output:
                        fc
        '''

        dim_feat = inp_feat.shape[1].value

        with tf.variable_scope(var_scope):

            fc_weights = tf.get_variable(name='fc_weights', dtype=tf.float32, initializer=tf.random_normal(shape=[dim_feat, num_of_outputs]))
            fc_bias    = tf.get_variable(name='fc_bias',    dtype=tf.float32, initializer=tf.zeros(shape=[num_of_outputs]))

            # tf.nn.xw_plus_b(x, weights, bias) = tf.matmul(x, weights) + biases
            fc = tf.nn.xw_plus_b(x=inp_feat, weights=fc_weights, biases=fc_bias)

            return fc





    def _fc2_multiLabel_classifier(self, inp_feat, label, num_of_outputs_fc1, num_of_outputs_fc2, name_scope):
        '''
            Function:
                        _fc2_multiLabel_classifier
            Input:
                        [1] inp_feat, dimension->[batch_size, dim_feat]
                        [2] label
                        [3] num_of_outputs_fc1
                        [4] num_of_outputs_fc2
                        [5] name_scope
            Output:
                        loss
        '''

        fc1 = self._fc_layer(inp_feat=inp_feat, num_of_outputs=num_of_outputs_fc1, var_scope='fc1')

        fc2 = self._fc_layer(inp_feat=fc1, num_of_outputs=num_of_outputs_fc2, var_scope='fc2')

        fc3 = self._fc_layer(inp_feat=fc2, num_of_outputs=self.num_of_class, var_scope='fc3')

        pred = tf.sigmoid(fc3)

        # tf.losses.sigmoid_cross_entropy( multi_class_labels,
        #                                  logits,
        #                                  weights=1.0,
        #                                  label_smoothing=0,
        #                                  scope=None,
        #                                  loss_collection=tf.GraphKeys.LOSSES,
        #                                  reduction=Reduction.SUM_BY_NONZERO_WEIGHTS
        #                                 )
        # new_multi_class_labels = multi_class_label * (1-label_smoothing) + 0.5*label_smoothing
        loss = tf.losses.log_loss(labels=self.label, predictions=pred)

        return loss



    def _build_model(self):

        # define '/data' and '/label' placeholder...
        self.rgb_feat   = tf.placeholder(dtype=tf.float32, shape=[self.batch_size, self.nframes, self.dim_rgb_feat])
        self.audio_feat = tf.placeholder(dtype=tf.float32, shape=[self.batch_size, self.nframes, self.dim_audio_feat])

        self.label = tf.placeholder(dtype=tf.float32, shape=[self.batch_size, self.num_of_class])


        # define embedding layer...
        emb_rgb   = self._embedding_layer(self.rgb_feat,   emb_size=256, name_scope='emb_rgb')
        emb_audio = self._embedding_layer(self.audio_feat, emb_size=256, name_scope='emb_audio')

        # define bi-directional-lstm layer...
        bidi_lstm_rgb   = self._bidirectional_LSTM_cell(inp_feat=emb_rgb, num_of_hidden_units=256, num_of_cell_state=self.nframes, var_scope='bidi_lstm_rgb')
        bidi_lstm_audio = self._bidirectional_LSTM_cell(inp_feat=emb_audio, num_of_hidden_units=256, num_of_cell_state=self.nframes, var_scope='bidi_lstm_audio')

        # define attention module...
        att_rgb   = self._attention_module(inp_feat=bidi_lstm_rgb, name_scope='att_rgb')
        att_audio = self._attention_module(inp_feat=bidi_lstm_audio, name_scope='att_audio')

        # feature concatenate i.e. 'rgb' + 'label'...
        concat_feat = tf.concat(values=[att_rgb, att_audio], axis=2)

        # define feature aggregation layer, i.e. simple sumPooling...
        aggt_feat = self._feature_aggregation_module(inp_feat=concat_feat)


        # define classifier and loss layer i.e. 2-layer fc + sigmoid + multi-label-crossEntropy ...
        self.loss = self._fc2_multiLabel_classifier(inp_feat=aggt_feat, label=self.label, num_of_outputs_fc1=4096, num_of_outputs_fc2=2048, name_scope='loss')






    def _train_phase(self):

        # !Load .tfrecord file...
        rgb_batch, audio_batch, labels_batch, videos_id_batch = self._load_youtube8M_tfrecord_data()

        # !Define training operation...
        self.train_op =  tf.train.GradientDescentOptimizer(self.base_lr_rate).minimize(self.loss)

        # !Intialize the global_variables and local_variables...
        self.sess.run(tf.global_variables_initializer())
        self.sess.run(tf.local_variables_initializer())


        # !Start-up Coordinator and QueueRunner...
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=self.sess, coord=coord)




        try:
            while(not coord.should_stop()):
                for iter in range(self.max_iteration):
                    # !Get mini-batch training data...
                    rgb, audio, label, video_id = self.sess.run([rgb_batch, audio_batch, labels_batch, videos_id_batch])

                    # !Execute training operations...
                    _, err = self.sess.run([self.train_op, self.loss], feed_dict={self.rgb_feat: rgb, self.audio_feat: audio, self.label:label})
                    print('!Loss of No.%d Iteration=%f' % (iter, err))

        except tf.errors.OutOfRangeError:
            print('!Frame-level Feature read completed...')
        finally:
            coord.request_stop()
            coord.join(threads=threads)


    # def _test_phase(self):

    # def _save_model(self):

    # def _load_model(self):

