# Created by babiking on Dec. 9th @tucodec...
#
# utils.py: function set of basic operations e.g. resize/dequantize

import tensorflow as tf



def _dequantize(inp_feat, max_quantized_value=2, min_quantized_value=-2):
    '''
        Function:
                    _dequantize, i.e. dequantize uint8 data into float32 format...
        Input:
                    [1] <tensor> inp_feat
                    [2] <int32>  max_quantized_value
                    [3] <int32>  min_quantized_value
        Output:
                    <tensor> out_feat
    '''


    assert max_quantized_value > min_quantized_value

    quantized_range = max_quantized_value - min_quantized_value

    scalar = quantized_range / 255.0
    bias = (quantized_range / 512.0) + min_quantized_value

    out_feat = inp_feat * scalar + bias

    return out_feat






def _resize_axis(inp_feat, axis, new_size, fill_value=0):
    '''
        Function:
                    _resize_axis,
                    i.e. padding or crop 'inp_feat' into 'new_size' along 'axis' direction with 'fill_value'...
        Input:
                    [1] <tensor>   inp_feat-> dimension: [nframes, dim_feat]
                    [2] <int32>    axis
                    [3] <float32>  fill_value
                    [4] <int32>    new_size
        Output:
                    <tensor> resized_feat

        note: referred from https://github.com/google/youtube-8m...

    '''


    # !Convert inp_feat into tensor...
    inp_feat = tf.convert_to_tensor(inp_feat)

    # !Get the shape of inp_feat along each dimension...
    shape_feat = tf.unstack(tf.shape(inp_feat))

    # !Calculate the padding_shape for inp_feat along 'axis' direction...
    pad_shape = shape_feat[:]
    pad_shape[axis] = tf.maximum(0, new_size-shape_feat[axis])
    pad_shape = tf.stack(pad_shape)

    # !Execute...
    # if (new_size > shape_feat[axis]):
    #           padding with fill_value
    # else:
    #           crop inp_feat[axis] into new_size
    shape_feat[axis] = tf.minimum(shape_feat[axis], new_size)
    shape_feat = tf.stack(shape_feat)




    # !Concanate inp_feat...
    resized_feat = tf.concat([tf.slice(inp_feat, tf.zeros_like(shape_feat), shape_feat),
                             tf.fill(pad_shape, tf.cast(fill_value, dtype=inp_feat.dtype))], axis=axis)

    # !Update shape of resize_feat...
    new_shape = inp_feat.get_shape().as_list()
    new_shape[axis] = new_size
    resized_feat.set_shape(new_shape)

    return resized_feat





