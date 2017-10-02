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


def pool2d(input_tensor, k_h=2, k_w=2, d_h=2, d_w=2):
    return tf.layers.max_pooling2d(inputs=input_tensor, pool_size=[k_h, k_w], strides=[d_h, d_w])


def upscale(input_tensor):
    shape_list = get_shape(input_tensor)[1:3]
    return tf.image.resize_nearest_neighbor(input_tensor, [shape_list[0]*2, shape_list[1]*2])


def conv_relu_batch_dropout(input_tensor, filters, is_training, drop_probability, name):
    with tf.variable_scope(name):
        conv = conv2d(input_tensor=input_tensor, filters=filters, activation=tf.nn.relu, name='conv')
        batch = tf.layers.batch_normalization(inputs=conv, training=is_training, name='batch')
        drop = tf.layers.dropout(inputs=batch, rate=drop_probability, training=is_training, name='drop')
    return drop


def dense_block(input_tensor, num_layers, growth_rate, is_training, drop_probability, name):
    with tf.variable_scope(name):
        concat = input_tensor
        for layer_idx in range(num_layers):
            layer_out = conv_relu_batch_dropout(input_tensor=concat, filters=growth_rate,
                                                name='layer{:d}'.format(layer_idx),
                                                is_training=is_training, drop_probability=drop_probability)
            concat = tf.concat([concat, layer_out], 3)
    return concat


def dense_block_down(input_tensor, num_layers, growth_rate, filters, name, is_training, drop_probability):
    with tf.variable_scope(name):
        db = dense_block(input_tensor=input_tensor, num_layers=num_layers, growth_rate=growth_rate,
                         is_training=is_training, drop_probability=drop_probability, name='db')
        conv = tf.layers.conv2d(inputs=db, filters=filters, kernel_size=[1, 1],
                                strides=[1, 1], padding='same', activation=tf.nn.relu, name='td')
        batch = tf.layers.batch_normalization(inputs=conv, training=is_training, name='batch')
        drop = tf.layers.dropout(inputs=batch, rate=drop_probability, training=is_training, name='drop')
    return drop


def simple_256(x, flags, drop_probability=0.0, is_training=False):
    """ stage-d1 256x256"""
    feature1 = dense_block_down(input_tensor=x, num_layers=4, growth_rate=16, filters=64,
                                name='feature_d1', is_training=is_training, drop_probability=drop_probability)
    """ stage-d2 128x128"""
    feature1_pool = pool2d(input_tensor=feature1)
    feature2 = dense_block_down(input_tensor=feature1_pool, num_layers=8, growth_rate=16, filters=128,
                                name='feature_d2', is_training=is_training, drop_probability=drop_probability)
    """ stage-d3 64x64"""
    feature2_pool = pool2d(input_tensor=feature2)
    feature3 = dense_block_down(input_tensor=feature2_pool, num_layers=16, growth_rate=16, filters=256,
                                name='feature_d3', is_training=is_training, drop_probability=drop_probability)

    # ------------------------------------------------------------------------------------------
    """ bottle_neck 32x32"""
    feature3_pool = pool2d(input_tensor=feature3)
    feature_bottle = dense_block_down(input_tensor=feature3_pool, num_layers=16, growth_rate=16, filters=512,
                                      is_training=is_training, drop_probability=drop_probability, name='feature_bottle')
    # ------------------------------------------------------------------------------------------

    """ stage-u3 64x64"""
    feature_up = upscale(feature_bottle)
    feature_concat = tf.concat([feature_up, feature3], 3)
    feature_up3 = dense_block_down(input_tensor=feature_concat, num_layers=16, growth_rate=16, filters=256,
                                   name='feature_up3', is_training=is_training, drop_probability=drop_probability)
    """ stage-u2 128x128"""
    feature_up = upscale(feature_up3)
    feature_concat = tf.concat([feature_up, feature2], 3)
    feature_up2 = dense_block_down(input_tensor=feature_concat, num_layers=8, growth_rate=16, filters=128,
                                   name='feature_up2', is_training=is_training, drop_probability=drop_probability)
    """ stage-u1 256x256"""
    feature_up = upscale(feature_up2)
    feature_concat = tf.concat([feature_up, feature1], 3)
    feature_up1 = dense_block_down(input_tensor=feature_concat, num_layers=4, growth_rate=16, filters=64,
                                   name='feature_up1', is_training=is_training, drop_probability=drop_probability)
    """ output 256x256"""
    output = tf.layers.conv2d(inputs=feature_up1, filters=flags.num_of_class, kernel_size=[3, 3],
                              strides=[1, 1], padding='same', activation=None, name='output')
    return output


def simple_256_pool(x, flags, drop_probability=0.0, is_training=False):
    """ stage-d1 256x256"""
    feature1 = dense_block_down(input_tensor=x, num_layers=4, growth_rate=16, filters=64,
                                name='feature_d1', is_training=is_training, drop_probability=drop_probability)
    """ stage-d2 128x128"""
    feature1_pool = pool2d(input_tensor=feature1)
    feature2 = dense_block_down(input_tensor=feature1_pool, num_layers=8, growth_rate=16, filters=128,
                                name='feature_d2', is_training=is_training, drop_probability=drop_probability)
    """ stage-d3 64x64"""
    feature2_pool = pool2d(input_tensor=feature2)
    feature3 = dense_block_down(input_tensor=feature2_pool, num_layers=16, growth_rate=16, filters=256,
                                name='feature_d3', is_training=is_training, drop_probability=drop_probability)

    # ------------------------------------------------------------------------------------------
    """ bottle_neck 32x32"""
    feature3_pool = pool2d(input_tensor=feature3)
    feature_bottle = dense_block_down(input_tensor=feature3_pool, num_layers=16, growth_rate=16, filters=512,
                                      is_training=is_training, drop_probability=drop_probability, name='feature_bottle')
    # ------------------------------------------------------------------------------------------

    """ stage-u3 64x64"""
    feature_up = upscale(feature_bottle)
    feature_concat = tf.concat([feature_up, feature2_pool], 3)
    feature_up3 = dense_block_down(input_tensor=feature_concat, num_layers=16, growth_rate=16, filters=256,
                                   name='feature_up3', is_training=is_training, drop_probability=drop_probability)
    """ stage-u2 128x128"""
    feature_up = upscale(feature_up3)
    feature_concat = tf.concat([feature_up, feature1_pool], 3)
    feature_up2 = dense_block_down(input_tensor=feature_concat, num_layers=8, growth_rate=16, filters=128,
                                   name='feature_up2', is_training=is_training, drop_probability=drop_probability)
    """ stage-u1 256x256"""
    feature_up = upscale(feature_up2)
    feature_concat = tf.concat([feature_up, x], 3)
    feature_up1 = dense_block_down(input_tensor=feature_concat, num_layers=4, growth_rate=16, filters=64,
                                   name='feature_up1', is_training=is_training, drop_probability=drop_probability)
    """ output 256x256"""
    output = tf.layers.conv2d(inputs=feature_up1, filters=flags.num_of_class, kernel_size=[3, 3],
                              strides=[1, 1], padding='same', activation=None, name='output')
    return output


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


def simple_ae2(x, flags, is_training=False):
    """ stage-1 400x400"""
    conv1_1 = conv2d(input_tensor=x, filters=16, activation=tf.nn.relu, name='conv1_1')
    batch_c1 = tf.layers.batch_normalization(inputs=conv1_1, training=is_training, name='batch_c1')
    """ stage-2 200x200"""
    pool1 = tf.layers.max_pooling2d(inputs=batch_c1, pool_size=[2, 2], strides=[2, 2])
    conv2_1 = conv2d(input_tensor=pool1, filters=32, activation=tf.nn.relu, name='conv2_1')
    batch_c2 = tf.layers.batch_normalization(inputs=conv2_1, training=is_training, name='batch_c2')
    """ stage-3 100x100"""
    pool2 = tf.layers.max_pooling2d(inputs=batch_c2, pool_size=[2, 2], strides=[2, 2])
    conv3_1 = conv2d(input_tensor=pool2, filters=64, activation=tf.nn.relu, name='conv3_1')
    conv3_2 = conv2d(input_tensor=conv3_1, filters=128, activation=tf.nn.relu, name='conv3_2')
    batch_c3 = tf.layers.batch_normalization(inputs=conv3_2, training=is_training, name='batch_c3')
    """ stage-4 50x50"""
    pool3 = tf.layers.max_pooling2d(inputs=batch_c3, pool_size=[2, 2], strides=[2, 2])
    conv4_1 = conv2d(input_tensor=pool3, filters=128, activation=tf.nn.relu, name='conv4_1')
    conv4_2 = conv2d(input_tensor=conv4_1, filters=256, activation=tf.nn.relu, name='conv4_2')
    batch_c4 = tf.layers.batch_normalization(inputs=conv4_2, training=is_training, name='batch_c4')

    ########################################################################################
    """ bottle neck 25x25"""
    pool4 = tf.layers.max_pooling2d(inputs=batch_c4, pool_size=[2, 2], strides=[2, 2])
    neck_1 = conv2d(input_tensor=pool4, filters=256, activation=tf.nn.relu, name='neck_1')
    neck_2 = conv2d(input_tensor=neck_1, filters=512, activation=tf.nn.relu, name='neck_2')
    batch_neck = tf.layers.batch_normalization(inputs=neck_2, training=is_training, name='batch_neck')
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