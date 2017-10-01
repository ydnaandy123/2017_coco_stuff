import tensorflow as tf


def get_shape(tensor):
    static_shape = tensor.shape.as_list()
    dynamic_shape = tf.unstack(tf.shape(tensor))
    dims = [s[1] if s[0] is None else s[0]
            for s in zip(static_shape, dynamic_shape)]
    return dims


def deconv2d(input_tensor, filters, k_h=3, k_w=3, d_h=2, d_w=2, activation=None, name="deconv2d"):
    deconv = tf.layers.conv2d_transpose(inputs=input_tensor, filters=filters, kernel_size=[k_h, k_w],
                                        kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                        strides=[d_h, d_w], padding='same', activation=activation, name=name)
    return deconv


def conv2d(input_tensor, filters, k_h=3, k_w=3, d_h=1, d_w=1, activation=None, name="conv2d"):
    conv = tf.layers.conv2d(inputs=input_tensor, filters=filters, kernel_size=[k_h, k_w],
                            kernel_initializer=tf.contrib.layers.xavier_initializer(),
                            strides=[d_h, d_w], padding='same', activation=activation, name=name)
    return conv


def simple_ae(x, flags, is_training=False):
    """ stage-1 400x400"""
    conv1 = conv2d(input_tensor=x, filters=32, activation=tf.nn.relu, name='conv1')
    batch_c1 = tf.layers.batch_normalization(inputs=conv1, training=is_training, name='batch_c1')
    """ stage-2 200x200"""
    pool1 = tf.layers.max_pooling2d(inputs=batch_c1, pool_size=[2, 2], strides=[2, 2])
    conv2 = conv2d(input_tensor=pool1, filters=64, activation=tf.nn.relu, name='conv2')
    batch_c2 = tf.layers.batch_normalization(inputs=conv2, training=is_training, name='batch_c2')
    """ stage-3 100x100"""
    pool2 = tf.layers.max_pooling2d(inputs=batch_c2, pool_size=[2, 2], strides=[2, 2])
    conv3 = conv2d(input_tensor=pool2, filters=128, activation=tf.nn.relu, name='conv3')
    batch_c3 = tf.layers.batch_normalization(inputs=conv3, training=is_training, name='batch_c3')
    """ stage-4 50x50"""
    pool3 = tf.layers.max_pooling2d(inputs=batch_c3, pool_size=[2, 2], strides=[2, 2])
    conv4 = conv2d(input_tensor=pool3, filters=256, activation=tf.nn.relu, name='conv4')
    batch_c4 = tf.layers.batch_normalization(inputs=conv4, training=is_training, name='batch_c4')

    ########################################################################################
    """ bottle neck 25x25"""
    pool4 = tf.layers.max_pooling2d(inputs=batch_c4, pool_size=[2, 2], strides=[2, 2])
    neck = conv2d(input_tensor=pool4, filters=512, activation=tf.nn.relu, name='neck')
    batch_neck = tf.layers.batch_normalization(inputs=neck, training=is_training, name='batch_neck')
    ########################################################################################

    """ stage-r4 50x50"""
    deconv4 = deconv2d(input_tensor=batch_neck, filters=256, activation=tf.nn.relu, name="deconv4")
    batch_d4 = tf.layers.batch_normalization(inputs=deconv4, training=is_training, name='batch_d4')
    concat4 = tf.concat([batch_d4, pool3], 3)
    """ stage-r3 100x100"""
    deconv3 = deconv2d(input_tensor=concat4, filters=128, activation=tf.nn.relu, name="deconv3")
    batch_d3 = tf.layers.batch_normalization(inputs=deconv3, training=is_training, name='batch_d3')
    concat3 = tf.concat([batch_d3, pool2], 3)
    """ stage-r2 200x200"""
    deconv2 = deconv2d(input_tensor=concat3, filters=64, activation=tf.nn.relu, name="deconv2")
    batch_d2 = tf.layers.batch_normalization(inputs=deconv2, training=is_training, name='batch_d2')
    concat2 = tf.concat([batch_d2, pool1], 3)
    """ stage-r1 400x400"""
    deconv1 = deconv2d(input_tensor=concat2, filters=32, activation=tf.nn.relu, name="deconv1")
    batch_d1 = tf.layers.batch_normalization(inputs=deconv1, training=is_training, name='batch_d1')
    concat1 = tf.concat([batch_d1, x], 3)
    """ output 400x400"""
    output = tf.layers.conv2d(inputs=concat1, filters=flags.num_of_class, kernel_size=[3, 3],
                              strides=[1, 1], padding='same', activation=None, name='output')
    return output