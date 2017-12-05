import tensorflow as tf
import numpy as np
from tensorflow.contrib.slim.python.slim.data import parallel_reader
from tensorflow.python.ops import parsing_ops


def _dequantize(input_feat, max_quantize_value=2, min_quantize_value=-2):
    '''
        Function:
                _dequantize
        Input:
                [1] input_feat: input feature map
                [2] max_quantize_value
                [3] min_quantize_value
        Output:
                output_feat: output_feature
    '''

    assert max_quantize_value > min_quantize_value
    quantize_range = max_quantize_value - min_quantize_value

    scalar = quantize_range / 255.0
    bias   = (quantize_range / 512.0) + min_quantize_value

    output_feat = input_feat * scalar + bias

    return output_feat









def _frame_padding(input_feat, padding_value=0, target_nframe=300):
    '''
        Function:
                _frame_padding
        Input:
                [1] <float> input_feat: nframe x dim_feat
                [2] <float> padding_value
                [3] <int>   target_nframe
        Output:
                output_feat
    '''
    current_nframe, dim_feat = input_feat.shape


    if ( current_nframe < target_nframe ):
        pad_patch = padding_value * np.ones(shape=[target_nframe - current_nframe, dim_feat])
        output_feat = np.concatenate((input_feat, pad_patch), axis=0)

        return output_feat
    else:
        return input_feat













def _convert_Youtube8M_tfrecord_to_numpy(tfrecord_filename):

    '''
        Function:
                _convert_Youtube8M_tfrecord_to_numpy
                i.e. parse each data_component according to example_prototxt
        Input:
                <string> tfrecord_filename
        Output:
                <dictionary> parsed_data
    '''

    reader = tf.TFRecordReader

    _, data = parallel_reader.parallel_read(data_sources=tfrecord_filename,
                                            reader_class=reader,
                                            num_epochs=1,
                                            num_readers=1,
                                            shuffle=False,
                                            capacity=256,
                                            min_after_dequeue=1)

    # build-up fileQueue and exampleQueue for tfrecords.file...
    context_feat, seq_feat = parsing_ops.parse_single_sequence_example(data,
                                                                       context_features={
                                                                           'video_id': tf.VarLenFeature(tf.string),
                                                                           'labels':   tf.VarLenFeature(tf.int64)
                                                                       },
                                                                       sequence_features={
                                                                           'rgb':   tf.FixedLenSequenceFeature([], tf.string),
                                                                           'audio': tf.FixedLenSequenceFeature([], tf.string)
                                                                       },
                                                                       example_name=" ")


    # standard framework for example parsing...
    with tf.Session() as sess:

        #--- initialize variables in tensorflow session ---#
        sess.run(tf.local_variables_initializer())
        sess.run(tf.global_variables_initializer())

        #--- start-up coordinator to manage the QueueRunner threads ---#
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        #--- training operations ---#
        try:
            total_rgb_feat   = []
            total_audio_feat = []
            total_label      = []

            while not coord.should_stop():

                video_context, video_features = sess.run((context_feat, seq_feat))

                #--- extract 'video_id' and 'labels' from context features ---#
                video_id = video_context['video_id'].values[0]
                labels   = video_context['labels'].values

                #--- one-hot vector for labels ---#
                labels = sess.run(tf.sparse_to_dense(labels, (4716, ), 1, validate_indices=False))

                #--- extract 'rgb' and 'audio' features from video features ---#
                hex_rgb_feat   = video_features['rgb']
                hex_audio_feat = video_features['audio']

                rgb_feat   = []
                audio_feat = []

                #--- convert hex data i.e. hex_rgb_feat to numpy.uint8 format ---#
                for ii in range(len(hex_rgb_feat)):
                    single_rgb_feat   = np.fromstring(hex_rgb_feat[ii],   dtype=np.uint8)
                    single_audio_feat = np.fromstring(hex_audio_feat[ii], dtype=np.uint8)

                    rgb_feat.append(single_rgb_feat)
                    audio_feat.append(single_audio_feat)



                #--- reshape e.g. [[1,2], [3,4]] -> [1,2; 3,4]
                rgb_feat = np.vstack(rgb_feat)
                audio_feat = np.vstack(audio_feat)



                #--- dequantize the rgb and audio features... ---#
                rgb_feat   = _dequantize(rgb_feat, 2, -2)
                audio_feat = _dequantize(audio_feat, 2, -2)



                #--- padding or crop to fixed nframe=300... ---#
                rgb_feat   = _frame_padding(input_feat=rgb_feat,   padding_value=0, target_nframe=300)
                audio_feat = _frame_padding(input_feat=audio_feat, padding_value=0, target_nframe=300)



                total_rgb_feat.append(rgb_feat)
                total_audio_feat.append(audio_feat)
                total_label.append(labels)

        except tf.errors.OutOfRangeError:
            print('!All video features have been exported...')
        finally:
            coord.request_stop()
            coord.join(threads=threads)

        return total_rgb_feat, total_audio_feat, total_label

    sess.close()



