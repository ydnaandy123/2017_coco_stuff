import tensorflow as tf
from tensorflow.contrib.layers import flatten


def flatten_fixed(input_tensor):
    # dim = tf.reduce_prod(tf.shape(input_tensor)[1:])
    # return tf.reshape(input_tensor, [-1, dim])
    return flatten(input_tensor)


def conv_layer(inputs, filters, kernel, stride=1, name="conv"):
    network = tf.layers.conv2d(
        inputs=inputs, filters=filters, kernel_size=kernel, strides=stride, padding='SAME', name=name)
    return network


def relu(input_tensor):
    return tf.nn.relu(input_tensor)


def batch_normalization(x, training, name='batch'):
    batch = tf.layers.batch_normalization(inputs=x, training=training, name=name)
    return batch


def drop_out(x, rate, training):
    return tf.layers.dropout(inputs=x, rate=rate, training=training)


def global_average_pooling(x, stride=1):
    # shape = get_shape(x)
    # return tf.layers.average_pooling2d(inputs=x, pool_size=shape[1:3], strides=stride)
    return tf.reduce_mean(input_tensor=x, axis=[1, 2], keep_dims=True)


def average_pooling(x, pool_size_h=2, pool_size_w=2, stride_h=2, stride_w=2, padding='VALID'):
    return tf.layers.average_pooling2d(
        inputs=x, pool_size=[pool_size_h, pool_size_w], strides=[stride_h, stride_w], padding=padding)


def max_pooling(x, pool_size_h=2, pool_size_w=2, stride_h=2, stride_w=2, padding='VALID'):
    return tf.layers.max_pooling2d(
        inputs=x, pool_size=[pool_size_h, pool_size_w], strides=[stride_h, stride_w], padding=padding)


def concatenation(layers):
    return tf.concat(layers, axis=3)


def linear(x, units):
    return tf.layers.dense(inputs=x, units=units, name='linear')


def get_shape(tensor):
    static_shape = tensor.shape.as_list()
    dynamic_shape = tf.unstack(tf.shape(tensor))
    dims = [s[1] if s[0] is None else s[0]
            for s in zip(static_shape, dynamic_shape)]
    return dims
