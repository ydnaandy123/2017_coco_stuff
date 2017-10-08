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


def upscale(input_tensor, s_w=2, s_h=2):
    shape_list = get_shape(input_tensor)[1:3]
    return tf.image.resize_nearest_neighbor(input_tensor, [shape_list[0]*s_w, shape_list[1]*s_h])


def upscale_blinear(input_tensor, s_w=2, s_h=2):
    shape_list = get_shape(input_tensor)[1:3]
    return tf.image.resize_bilinear(input_tensor, [shape_list[0]*s_w, shape_list[1]*s_h])


def conv_relu_batch_dropout(input_tensor, filters, is_training, drop_probability, name):
    with tf.variable_scope(name):
        conv = conv2d(input_tensor=input_tensor, filters=filters, activation=tf.nn.relu, name='conv')
        batch = tf.layers.batch_normalization(inputs=conv, training=is_training, name='batch')
        drop = tf.layers.dropout(inputs=batch, rate=drop_probability, training=is_training, name='drop')
    return drop


def conv_relu_batch(input_tensor, filters, is_training, name):
    with tf.variable_scope(name):
        conv = conv2d(input_tensor=input_tensor, filters=filters, activation=tf.nn.relu, name='conv')
        batch = tf.layers.batch_normalization(inputs=conv, training=is_training, name='batch')
    return batch


def conv_relu_batch_dropout_dilated(input_tensor, filters, is_training, drop_probability, name, dilation_rate=(1, 1)):
    with tf.variable_scope(name):
        conv = tf.layers.conv2d(inputs=input_tensor, filters=filters, kernel_size=[3, 3], dilation_rate=dilation_rate,
                                kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                strides=[1, 1], padding='same', activation=tf.nn.relu, name='conv')
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
                                strides=[1, 1], padding='same', activation=tf.nn.relu, name='down')
        batch = tf.layers.batch_normalization(inputs=conv, training=is_training, name='batch')
        drop = tf.layers.dropout(inputs=batch, rate=drop_probability, training=is_training, name='drop')
    return drop


def dilated_layer(input_tensor, filters, is_training, drop_probability, name):
    with tf.variable_scope(name):
        conv = tf.layers.conv2d(inputs=input_tensor, filters=filters, kernel_size=[3, 3],
                                kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                dilation_rate=[2, 2],
                                strides=[1, 1], padding='same', activation=tf.nn.relu, name=name)
        batch = tf.layers.batch_normalization(inputs=conv, training=is_training, name='batch')
        drop = tf.layers.dropout(inputs=batch, rate=drop_probability, training=is_training, name='drop')
    return drop


def stacked_layer(input_tensor, is_training=False, drop_probability=0.0, name='stacked'):
    with tf.variable_scope(name):
        """ td3 64x64"""
        feature3 = dense_block_down(input_tensor=input_tensor, num_layers=8, growth_rate=12, filters=64,
                                    name='feature_td3', is_training=is_training, drop_probability=drop_probability)
        """ td4 32x32"""
        feature3_pool = pool2d(input_tensor=feature3)
        feature4 = dense_block_down(input_tensor=feature3_pool, num_layers=8, growth_rate=12, filters=128,
                                    name='feature_td4', is_training=is_training, drop_probability=drop_probability)
        """ td5 16x16"""
        feature4_pool = pool2d(input_tensor=feature4)
        feature5 = dense_block_down(input_tensor=feature4_pool, num_layers=12, growth_rate=12, filters=256,
                                    name='feature_td5', is_training=is_training, drop_probability=drop_probability)
        # ------------------------------------------------------------------------------------------
        """ bottle_neck 8x8"""
        feature5_pool = pool2d(input_tensor=feature5)
        feature_bottle = dense_block_down(input_tensor=feature5_pool, num_layers=16, growth_rate=12, filters=512,
                                          is_training=is_training, drop_probability=drop_probability,
                                          name='feature_bottle')
        # ------------------------------------------------------------------------------------------
        """ tu5 16x16"""
        feature_up = upscale(feature_bottle)
        feature_concat = tf.concat([feature_up, feature4_pool], 3)
        feature_up5 = dense_block_down(input_tensor=feature_concat, num_layers=8, growth_rate=12, filters=256,
                                       name='feature_up5', is_training=is_training, drop_probability=drop_probability)
        """ tu4 32est-devx32"""
        feature_up = upscale(feature_up5)
        feature_concat = tf.concat([feature_up, feature3_pool], 3)
        feature_up4 = dense_block_down(input_tensor=feature_concat, num_layers=12, growth_rate=12, filters=128,
                                       name='feature_up4', is_training=is_training, drop_probability=drop_probability)
        """ tu3 64x64"""
        feature_up = upscale(feature_up4)
        feature_concat = tf.concat([feature_up, input_tensor], 3)
        feature_up3 = dense_block_down(input_tensor=feature_concat, num_layers=8, growth_rate=12, filters=64,
                                       name='feature_up3', is_training=is_training, drop_probability=drop_probability)
    return feature_up3


def stuff512(x, flags, drop_probability=0.0, is_training=False):
    """ bilinear down-up 512x512"""
    x_down = tf.image.resize_bilinear(x, (get_shape(x)[1] // 2, get_shape(x)[2] // 2))
    x_down = (x_down / 127.5) - 1.0
    # ------------------------------------------------------------------------------------------
    # Use conv instead of resize to learn the useful features
    """ td1 256x256"""
    feature1 = dense_block_down(input_tensor=x_down, num_layers=4, growth_rate=8, filters=32,
                                name='feature_td1', is_training=is_training, drop_probability=drop_probability)
    """ td2 128x128"""
    feature1_pool = pool2d(input_tensor=feature1)
    feature2 = dense_block_down(input_tensor=feature1_pool, num_layers=8, growth_rate=8, filters=64,
                                name='feature_td2', is_training=is_training, drop_probability=drop_probability)
    # ------------------------------------------------------------------------------------------
    """ stacked-1 64x64"""
    feature2_pool = pool2d(input_tensor=feature2)
    stacked1 = stacked_layer(input_tensor=feature2_pool, is_training=is_training,
                             drop_probability=drop_probability, name='stacked-1')
    stacked2 = stacked_layer(input_tensor=stacked1, is_training=is_training,
                             drop_probability=drop_probability, name='stacked-2')
    # feature_up4 = feature_up3
    """ output 64x64"""
    output_stacked1 = tf.layers.conv2d(inputs=stacked1, filters=flags.num_of_class, kernel_size=[1, 1],
                                       strides=[1, 1], padding='same', activation=None, name='output_stacked1')
    output_stacked2 = tf.layers.conv2d(inputs=stacked2, filters=flags.num_of_class, kernel_size=[1, 1],
                                       strides=[1, 1], padding='same', activation=None, name='output_stacked2')
    # output_stacked2 = output_stacked1
    """ bilinear upscale-up 512x512"""
    # output_up = tf.image.resize_nearest_neighbor(output_stacked2, get_shape(x)[1:3])
    output_up = tf.image.resize_bilinear(output_stacked2, get_shape(x)[1:3])

    return [output_stacked1, output_stacked2],  output_up


def dense_ae(x, flags, drop_probability=0.0, is_training=False):
    """ Original size 448x448"""
    """ bilinear scale-down 224x224"""
    x_down = tf.image.resize_bilinear(x, (get_shape(x)[1] // 2, get_shape(x)[2] // 2))
    x_down = (x_down / 127.5) - 1.0
    """ td1 224x224"""
    down_1 = conv_relu_batch_dropout(input_tensor=x_down, filters=32, name='down_1',
                                     is_training=is_training, drop_probability=drop_probability)
    # ------------------------------------------------------------------------------------------
    # Use conv instead of resize to learn the useful features
    """ td2 112x112"""
    pool_1 = pool2d(down_1)
    down_2 = conv_relu_batch_dropout(input_tensor=pool_1, filters=64, name='down_2',
                                     is_training=is_training, drop_probability=drop_probability)
    """ td3 56x56"""
    pool_2 = pool2d(down_2)
    down_3 = conv_relu_batch_dropout(input_tensor=pool_2, filters=128, name='down_3',
                                     is_training=is_training, drop_probability=drop_probability)
    """ td4 28x28"""
    pool_3 = pool2d(down_3)
    down_4 = conv_relu_batch_dropout(input_tensor=pool_3, filters=256, name='down_4',
                                     is_training=is_training, drop_probability=drop_probability)
    """ td5 14x14"""
    pool_4 = pool2d(down_4)
    down_5 = conv_relu_batch_dropout(input_tensor=pool_4, filters=512, name='down_5',
                                     is_training=is_training, drop_probability=drop_probability)

    """ bottle_neck 7x7"""
    pool_5 = pool2d(down_5)
    bottle_neck = conv_relu_batch_dropout(input_tensor=pool_5, filters=1024, name='bottle_neck',
                                          is_training=is_training, drop_probability=drop_probability)

    """ tu5 14x14"""
    deconv5 = deconv2d(input_tensor=bottle_neck, filters=512, activation=tf.nn.relu, name="deconv_5")
    batch_d5 = tf.layers.batch_normalization(inputs=deconv5, training=is_training, name='batch_d5')
    drop_d5 = tf.layers.dropout(inputs=batch_d5, rate=drop_probability, training=is_training, name='drop_5')
    concat5 = tf.concat([drop_d5, pool_4], 3)

    """ tu4 28x28"""
    deconv4 = deconv2d(input_tensor=concat5, filters=256, activation=tf.nn.relu, name="deconv_4")
    batch_d4 = tf.layers.batch_normalization(inputs=deconv4, training=is_training, name='batch_d4')
    drop_d4 = tf.layers.dropout(inputs=batch_d4, rate=drop_probability, training=is_training, name='drop_4')
    concat4 = tf.concat([drop_d4, pool_3], 3)

    """ tu3 56x56"""
    deconv3 = deconv2d(input_tensor=concat4, filters=128, activation=tf.nn.relu, name="deconv_3")
    batch_d3 = tf.layers.batch_normalization(inputs=deconv3, training=is_training, name='batch_d3')
    drop_d3 = tf.layers.dropout(inputs=batch_d3, rate=drop_probability, training=is_training, name='drop_3')
    concat3 = tf.concat([drop_d3, pool_2], 3)

    # feature_up4 = feature_up3
    """ output 64x64"""
    output_stacked1 = tf.layers.conv2d(inputs=concat3, filters=flags.num_of_class, kernel_size=[1, 1],
                                       strides=[1, 1], padding='same', activation=None, name='output_stacked1')
    output_stacked2 = tf.layers.conv2d(inputs=concat3, filters=flags.num_of_class, kernel_size=[1, 1],
                                       strides=[1, 1], padding='same', activation=None, name='output_stacked2')
    # output_stacked2 = output_stacked1
    """ bilinear upscale-up 512x512"""
    # output_up = tf.image.resize_nearest_neighbor(output_stacked2, get_shape(x)[1:3])
    output_up = tf.image.resize_bilinear(output_stacked2, get_shape(x)[1:3])

    return [output_stacked1, output_stacked2],  output_up


def deconv_relu_batch_dropout(input_tensor, filters, is_training, drop_probability, name):
    with tf.variable_scope(name):
        deconv = deconv2d(input_tensor=input_tensor, filters=filters, activation=tf.nn.relu, name="deconv")
        batch = tf.layers.batch_normalization(inputs=deconv, training=is_training, name='batch')
        drop = tf.layers.dropout(inputs=batch, rate=drop_probability, training=is_training, name='drop')
    return drop


def ae_stack_layer(input_tensor, name, drop_probability, is_training):
    with tf.variable_scope(name):
        """ td3 56x56"""
        down_3 = conv_relu_batch_dropout(input_tensor=input_tensor, filters=128, name='down_3',
                                         is_training=is_training, drop_probability=drop_probability)
        """ td4 28x28"""
        pool_3 = pool2d(down_3)
        down_4 = conv_relu_batch_dropout(input_tensor=pool_3, filters=256, name='down_4',
                                         is_training=is_training, drop_probability=drop_probability)
        """ td5 14x14"""
        pool_4 = pool2d(down_4)
        down_5 = conv_relu_batch_dropout(input_tensor=pool_4, filters=512, name='down_5',
                                         is_training=is_training, drop_probability=drop_probability)

        """ bottle_neck 7x7"""
        pool_5 = pool2d(down_5)
        bottle_neck = conv_relu_batch_dropout(input_tensor=pool_5, filters=1024, name='bottle_neck',
                                              is_training=is_training, drop_probability=drop_probability)

        # ------------------------------------------------------------------------------------------
        """ tu5 14x14"""
        feature_up = upscale(bottle_neck)
        feature_concat = tf.concat([feature_up, pool_4], 3)
        feature_up5 = conv_relu_batch_dropout(input_tensor=feature_concat, filters=512, name='up_5',
                                              is_training=is_training, drop_probability=drop_probability)
        """ tu4 28x28"""
        feature_up = upscale(feature_up5)
        feature_concat = tf.concat([feature_up, pool_3], 3)
        feature_up4 = conv_relu_batch_dropout(input_tensor=feature_concat, filters=256, name='up_4',
                                              is_training=is_training, drop_probability=drop_probability)
        """ tu3 56x56"""
        feature_up = upscale(feature_up4)
        feature_concat = tf.concat([feature_up, input_tensor], 3)
        feature_up3 = conv_relu_batch_dropout(input_tensor=feature_concat, filters=128, name='up_3',
                                              is_training=is_training, drop_probability=drop_probability)
    return feature_up3


def ae_stack_layer_stride(input_tensor, name, drop_probability, is_training):
    with tf.variable_scope(name):
        """ td3 56x56"""
        down_3 = conv_relu_batch_dropout(input_tensor=input_tensor, filters=128, name='down_3',
                                         is_training=is_training, drop_probability=drop_probability)
        """ td4 28x28"""
        pool_3 = pool2d(down_3)
        down_4 = conv_relu_batch_dropout(input_tensor=pool_3, filters=256, name='down_4',
                                         is_training=is_training, drop_probability=drop_probability)
        """ td5 14x14"""
        pool_4 = pool2d(down_4)
        down_5 = conv_relu_batch_dropout(input_tensor=pool_4, filters=512, name='down_5',
                                         is_training=is_training, drop_probability=drop_probability)

        """ bottle_neck 7x7"""
        pool_5 = pool2d(down_5)
        bottle_neck = conv_relu_batch_dropout(input_tensor=pool_5, filters=1024, name='bottle_neck',
                                              is_training=is_training, drop_probability=drop_probability)

        # ------------------------------------------------------------------------------------------
        """ tu5 14x14"""
        up_5 = deconv_relu_batch_dropout(input_tensor=bottle_neck, filters=512, name='up_5',
                                         is_training=is_training, drop_probability=drop_probability)
        concat_5 = tf.concat([up_5, pool_4], 3)
        """ tu4 28x28"""
        up_4 = deconv_relu_batch_dropout(input_tensor=concat_5, filters=256, name='up_4',
                                         is_training=is_training, drop_probability=drop_probability)
        concat_4 = tf.concat([up_4, pool_3], 3)
        """ tu3 56x56"""
        up_3 = deconv_relu_batch_dropout(input_tensor=concat_4, filters=128, name='up_3',
                                         is_training=is_training, drop_probability=drop_probability)
        concat_3 = tf.concat([up_3, input_tensor], 3)

    return concat_3


def stacked_layer_stride(input_tensor, is_training=False, drop_probability=0.0, name='stacked'):
    with tf.variable_scope(name):
        """ td3 64x64"""
        feature3 = dense_block_down(input_tensor=input_tensor, num_layers=4, growth_rate=16, filters=128,
                                    name='feature_td3', is_training=is_training, drop_probability=drop_probability)
        """ td4 32x32"""
        feature3_pool = pool2d(input_tensor=feature3)
        feature4 = dense_block_down(input_tensor=feature3_pool, num_layers=8, growth_rate=16, filters=256,
                                    name='feature_td4', is_training=is_training, drop_probability=drop_probability)
        """ td5 16x16"""
        feature4_pool = pool2d(input_tensor=feature4)
        feature5 = dense_block_down(input_tensor=feature4_pool, num_layers=8, growth_rate=32, filters=512,
                                    name='feature_td5', is_training=is_training, drop_probability=drop_probability)
        # ------------------------------------------------------------------------------------------
        """ bottle_neck 8x8"""
        feature5_pool = pool2d(input_tensor=feature5)
        feature_bottle = dense_block_down(input_tensor=feature5_pool, num_layers=12, growth_rate=32, filters=1024,
                                          is_training=is_training, drop_probability=drop_probability,
                                          name='feature_bottle')
        # ------------------------------------------------------------------------------------------
        """ tu5 16x16"""
        feature_up = deconv_relu_batch_dropout(input_tensor=feature_bottle, filters=512, name='up_5',
                                               is_training=is_training, drop_probability=drop_probability)
        feature_concat = tf.concat([feature_up, feature4_pool], 3)
        """ tu4 32est-devx32"""
        feature_up = deconv_relu_batch_dropout(input_tensor=feature_up5, filters=512, name='up_4',
                                               is_training=is_training, drop_probability=drop_probability)
        feature_concat = tf.concat([feature_up, feature3_pool], 3)
        feature_up4 = dense_block_down(input_tensor=feature_concat, num_layers=8, growth_rate=16, filters=256,
                                       name='feature_up4', is_training=is_training, drop_probability=drop_probability)
        """ tu3 64x64"""
        feature_up = deconv_relu_batch_dropout(input_tensor=feature_up4, filters=512, name='up_3',
                                               is_training=is_training, drop_probability=drop_probability)
        feature_concat = tf.concat([feature_up, input_tensor], 3)
        feature_up3 = dense_block_down(input_tensor=feature_concat, num_layers=4, growth_rate=16, filters=128,
                                       name='feature_up3', is_training=is_training, drop_probability=drop_probability)
    return feature_up3


def ae_stack_layer_class(input_tensor, name, drop_probability, is_training):
    with tf.variable_scope(name):
        """ td2 112x112"""
        down_2 = conv_relu_batch_dropout(input_tensor=input_tensor, filters=64, name='down_2',
                                         is_training=is_training, drop_probability=drop_probability)
        """ td3 56x56"""
        pool_2 = pool2d(down_2)
        down_3 = conv_relu_batch_dropout(input_tensor=pool_2, filters=128, name='down_3',
                                         is_training=is_training, drop_probability=drop_probability)
        """ td4 28x28"""
        pool_3 = pool2d(down_3)
        # down_4 = conv_relu_batch_dropout(input_tensor=pool_3, filters=256, name='down_4',
        #                                  is_training=is_training, drop_probability=drop_probability)
        down_4 = dense_block_down(input_tensor=pool_3, num_layers=4, growth_rate=64, filters=256,
                                  name='down_4', is_training=is_training, drop_probability=drop_probability)
        """ td5 14x14"""
        pool_4 = pool2d(down_4)
        # down_5 = conv_relu_batch_dropout(input_tensor=pool_4, filters=512, name='down_5',
        #                                  is_training=is_training, drop_probability=drop_probability)
        down_5 = dense_block_down(input_tensor=pool_4, num_layers=4, growth_rate=128, filters=512,
                                  name='down_5', is_training=is_training, drop_probability=drop_probability)

        """ bottle_neck 7x7"""
        pool_5 = pool2d(down_5)
        # bottle_neck = conv_relu_batch_dropout(input_tensor=pool_5, filters=1024, name='bottle_neck',
        #                                       is_training=is_training, drop_probability=drop_probability)
        bottle_neck = dense_block_down(input_tensor=pool_5, num_layers=8, growth_rate=128, filters=1024,
                                       name='bottle_neck', is_training=is_training, drop_probability=drop_probability)

        # ------------------------------------------------------------------------------------------
        """ tu5 14x14"""
        feature_up = upscale(bottle_neck)
        feature_concat = tf.concat([feature_up, pool_4], 3)
        feature_up5 = dense_block_down(input_tensor=feature_concat, num_layers=4, growth_rate=128, filters=512,
                                       name='up_5', is_training=is_training, drop_probability=drop_probability)
        """ tu4 28x28"""
        feature_up = upscale(feature_up5)
        feature_concat = tf.concat([feature_up, pool_3], 3)
        feature_up4 = dense_block_down(input_tensor=feature_concat, num_layers=4, growth_rate=64, filters=256,
                                       name='up_4', is_training=is_training, drop_probability=drop_probability)
        """ tu3 56x56"""
        feature_up = upscale(feature_up4)
        feature_concat = tf.concat([feature_up, pool_2], 3)
        feature_up3 = conv_relu_batch_dropout(input_tensor=feature_concat, filters=128, name='up_3',
                                              is_training=is_training, drop_probability=drop_probability)
        """ tu2 112x112"""
        feature_up = upscale(feature_up3)
        feature_concat = tf.concat([feature_up, input_tensor], 3)
        feature_up2 = conv_relu_batch_dropout(input_tensor=feature_concat, filters=64, name='up_2',
                                              is_training=is_training, drop_probability=drop_probability)
    return feature_up2


def class_super(x, flags, drop_probability=0.0, is_training=False):
    """ Original size 448x448"""
    """ bilinear scale-down 224x224 (data can be augmented with pretrain denseNet)"""
    x_down = tf.image.resize_bilinear(x, (get_shape(x)[1] // 2, get_shape(x)[2] // 2))
    x_down = (x_down / 127.5) - 1.0
    # ------------------------------------------------------------------------------------------
    # Use conv instead of resize to learn the useful features
    """ td1 224x224"""
    feature1 = dense_block_down(input_tensor=x_down, num_layers=4, growth_rate=8, filters=32,
                                name='feature_td1', is_training=is_training, drop_probability=drop_probability)
    """ td2 112x112"""
    feature1_pool = pool2d(input_tensor=feature1)

    stack_feature1 = ae_stack_layer_class(input_tensor=feature1_pool, name='stack1',
                                          drop_probability=drop_probability, is_training=is_training)
    output_stacked1 = tf.layers.conv2d(inputs=stack_feature1, filters=3, activation=None,
                                       kernel_size=[1, 1], strides=[1, 1], padding='same', name='output_stack1')

    stack_input2 = tf.concat([feature1_pool, output_stacked1], 3)
    stack_feature2 = ae_stack_layer_class(input_tensor=stack_input2, name='stack2',
                                          drop_probability=drop_probability, is_training=is_training)
    output_stacked2 = tf.layers.conv2d(inputs=stack_feature2, filters=16, activation=None,
                                       kernel_size=[1, 1], strides=[1, 1], padding='same', name='output_stack2')

    stack_input3 = tf.concat([feature1_pool, output_stacked2], 3)
    stack_feature3 = ae_stack_layer_class(input_tensor=stack_input3, name='stack3',
                                          drop_probability=drop_probability, is_training=is_training)
    output_stacked3 = tf.layers.conv2d(inputs=stack_feature3, filters=92, activation=None,
                                       kernel_size=[1, 1], strides=[1, 1], padding='same', name='output_stack3')

    """ bilinear upscale-up 512x512"""
    output_up = tf.image.resize_bilinear(output_stacked3, get_shape(x)[1:3])

    return [output_stacked1, output_stacked2, output_stacked3], output_up


def ae_stack_layer_class_lite(input_tensor, name, drop_probability, is_training):
    with tf.variable_scope(name):
        """ td2 112x112"""
        """ td3 56x56"""
        pool_2 = pool2d(input_tensor)
        down_3 = conv_relu_batch_dropout(input_tensor=pool_2, filters=256, name='down_3',
                                         is_training=is_training, drop_probability=drop_probability)
        """ td4 28x28"""
        pool_3 = pool2d(down_3)
        down_4 = conv_relu_batch_dropout(input_tensor=pool_3, filters=512, name='down_4',
                                         is_training=is_training, drop_probability=drop_probability)
        # down_4 = dense_block_down(input_tensor=pool_3, num_layers=4, growth_rate=64, filters=256,
        #                           name='down_4', is_training=is_training, drop_probability=drop_probability)
        """ td5 14x14"""
        pool_4 = pool2d(down_4)
        down_5 = conv_relu_batch_dropout(input_tensor=pool_4, filters=512, name='down_5',
                                         is_training=is_training, drop_probability=drop_probability)
        # down_5 = dense_block_down(input_tensor=pool_4, num_layers=4, growth_rate=128, filters=512,
        #                           name='down_5', is_training=is_training, drop_probability=drop_probability)

        """ bottle_neck 7x7"""
        pool_5 = pool2d(down_5)
        bottle_neck_1 = conv_relu_batch_dropout_dilated(input_tensor=pool_5, filters=1024, name='bottle_neck_1',
                                                        is_training=is_training, drop_probability=drop_probability,
                                                        dilation_rate=(1, 1))
        bottle_neck_2 = conv_relu_batch_dropout_dilated(input_tensor=bottle_neck_1, filters=1024, name='bottle_neck_2',
                                                        is_training=is_training, drop_probability=drop_probability,
                                                        dilation_rate=(2, 2))
        bottle_neck_3 = conv_relu_batch_dropout_dilated(input_tensor=bottle_neck_2, filters=1024, name='bottle_neck_3',
                                                        is_training=is_training, drop_probability=drop_probability,
                                                        dilation_rate=(3, 3))
        # bottle_neck = dense_block_down(input_tensor=pool_5, num_layers=8, growth_rate=128, filters=1024,
        #                                name='bottle_neck', is_training=is_training, drop_probability=drop_probability)

        # ------------------------------------------------------------------------------------------
        """ tu5 14x14"""
        feature_up = upscale(bottle_neck_3)
        feature_concat = tf.concat([feature_up, down_5], 3)
        feature_up5 = conv_relu_batch_dropout(input_tensor=feature_concat, filters=512, name='up_5',
                                              is_training=is_training, drop_probability=drop_probability)
        # feature_up5 = dense_block_down(input_tensor=feature_concat, num_layers=4, growth_rate=128, filters=512,
        #                                name='up_5', is_training=is_training, drop_probability=drop_probability)
        """ tu4 28x28"""
        feature_up = upscale(feature_up5)
        feature_concat = tf.concat([feature_up, down_4], 3)
        feature_up4 = conv_relu_batch_dropout(input_tensor=feature_concat, filters=512, name='up_4',
                                              is_training=is_training, drop_probability=drop_probability)
        # feature_up4 = dense_block_down(input_tensor=feature_concat, num_layers=4, growth_rate=64, filters=256,
        #                                name='up_4', is_training=is_training, drop_probability=drop_probability)
        """ tu3 56x56"""
        feature_up = upscale(feature_up4)
        feature_concat = tf.concat([feature_up, down_3], 3)
        feature_up3 = conv_relu_batch_dropout(input_tensor=feature_concat, filters=256, name='up_3',
                                              is_training=is_training, drop_probability=drop_probability)
        # feature_up3 = conv_relu_batch_dropout(input_tensor=feature_concat, filters=128, name='up_3',
        #                                       is_training=is_training, drop_probability=drop_probability)
        """ tu2 112x112"""
        feature_up = upscale(feature_up3)
        feature_concat = tf.concat([feature_up, input_tensor], 3)
        feature_up2 = conv_relu_batch_dropout(input_tensor=feature_concat, filters=128, name='up_2',
                                              is_training=is_training, drop_probability=drop_probability)
        # feature_up2 = conv_relu_batch_dropout(input_tensor=feature_concat, filters=64, name='up_2',
        #                                       is_training=is_training, drop_probability=drop_probability)
    return feature_up2


def class_super_lite(x, flags, drop_probability=0.0, is_training=False):
    """ Original size 448x448"""
    """ bilinear scale-down 224x224 (data can be augmented with pretrain denseNet)"""
    # x_down = tf.image.resize_bilinear(x, (get_shape(x)[1] // 2, get_shape(x)[2] // 2))
    x_normal = (x / 127.5) - 1.0
    # ------------------------------------------------------------------------------------------
    # Use conv instead of resize to learn the useful features
    feature0 = conv_relu_batch_dropout(input_tensor=x_normal, filters=32, name='feature_td0',
                                       is_training=is_training, drop_probability=drop_probability)
    """ td1 224x224"""
    feature0_pool = pool2d(input_tensor=feature0)
    feature1 = conv_relu_batch_dropout(input_tensor=feature0_pool, filters=64, name='feature_td1',
                                       is_training=is_training, drop_probability=drop_probability)
    """ td2 112x112"""
    feature1_pool = pool2d(input_tensor=feature1)
    feature2 = conv_relu_batch_dropout(input_tensor=feature1_pool, filters=128, name='feature_td2',
                                       is_training=is_training, drop_probability=drop_probability)
    """ stacked AE 112x112"""
    stack_feature1 = ae_stack_layer_class_lite(input_tensor=feature2, name='stack1',
                                               drop_probability=drop_probability, is_training=is_training)

    stack_input2 = tf.concat([feature2, stack_feature1], 3)
    stack_feature2 = ae_stack_layer_class_lite(input_tensor=stack_input2, name='stack2',
                                               drop_probability=drop_probability, is_training=is_training)

    stack_input3 = tf.concat([feature2, stack_feature2], 3)
    stack_feature3 = ae_stack_layer_class_lite(input_tensor=stack_input3, name='stack3',
                                               drop_probability=drop_probability, is_training=is_training)

    """ output branch 224x224"""
    stack_feature1_up = upscale(stack_feature1)
    stack_feature1_concat = tf.concat([stack_feature1_up, feature1], 3)
    stack_feature1_conv = conv_relu_batch_dropout(input_tensor=stack_feature1_concat, filters=64, name='stack1_conv',
                                                  is_training=is_training, drop_probability=drop_probability)
    output_stacked1 = tf.layers.conv2d(inputs=stack_feature1_conv, filters=3, activation=None,
                                       kernel_size=[1, 1], strides=[1, 1], padding='same', name='output_stack1')

    stack_feature2_up = upscale(stack_feature2)
    stack_feature2_concat = tf.concat([stack_feature2_up, feature1], 3)
    stack_feature2_conv = conv_relu_batch_dropout(input_tensor=stack_feature2_concat, filters=64, name='stack2_conv',
                                                  is_training=is_training, drop_probability=drop_probability)
    output_stacked2 = tf.layers.conv2d(inputs=stack_feature2_conv, filters=16, activation=None,
                                       kernel_size=[1, 1], strides=[1, 1], padding='same', name='output_stack2')

    stack_feature3_up = upscale(stack_feature3)
    stack_feature3_concat = tf.concat([stack_feature3_up, feature1], 3)
    stack_feature3_conv = conv_relu_batch_dropout(input_tensor=stack_feature3_concat, filters=64, name='stack3_conv',
                                                  is_training=is_training, drop_probability=drop_probability)
    output_stacked3 = tf.layers.conv2d(inputs=stack_feature3_conv, filters=92, activation=None,
                                       kernel_size=[1, 1], strides=[1, 1], padding='same', name='output_stack3')
    """ bilinear upscale-up 448x448"""
    output_up = tf.image.resize_bilinear(output_stacked3, get_shape(x)[1:3])

    return [output_stacked1, output_stacked2, output_stacked3], output_up


def batch_conv_relu_dropout_dilated(input_tensor, filters, is_training, drop_probability, name, dilation_rate=(1, 1)):
    with tf.variable_scope(name):
        batch = tf.layers.batch_normalization(inputs=input_tensor, training=is_training, name='batch')
        conv = tf.layers.conv2d(inputs=batch, filters=filters, kernel_size=[3, 3], dilation_rate=dilation_rate,
                                kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                strides=[1, 1], padding='same', activation=tf.nn.relu, name='conv')
        drop = tf.layers.dropout(inputs=conv, rate=drop_probability, training=is_training, name='drop')
    return drop


def upscale_concat_batch_conv_relu(input_tensor, concate_tensor, filters, is_training, name):
    with tf.variable_scope(name):
        feature_up = upscale(input_tensor)
        feature_concat = tf.concat([feature_up, concate_tensor], 3)
        batch = tf.layers.batch_normalization(inputs=feature_concat, training=is_training, name='batch')
        conv = conv2d(input_tensor=batch, filters=filters, activation=tf.nn.relu, name='conv')
    return conv


def ae_multi_stack(input_tensor, name, drop_probability, is_training):
    with tf.variable_scope(name):
        """ td3 56x56"""
        td3 = pooling_batch_conv_relu(input_tensor=input_tensor, filters=256, name='td3', is_training=is_training)
        """ td4 28x28"""
        td4 = pooling_batch_conv_relu(input_tensor=td3, filters=512, name='td4', is_training=is_training)
        """ td5 14x14"""
        td5 = pooling_batch_conv_relu(input_tensor=td4, filters=512, name='td5', is_training=is_training)

        """ bottle_neck 7x7"""
        pool_5 = pool2d(td5)
        bottle_neck_1 = batch_conv_relu_dropout_dilated(input_tensor=pool_5, filters=1024, name='bottle_neck_1',
                                                        is_training=is_training, drop_probability=drop_probability,
                                                        dilation_rate=(1, 1))
        bottle_neck_2 = batch_conv_relu_dropout_dilated(input_tensor=bottle_neck_1, filters=1024, name='bottle_neck_2',
                                                        is_training=is_training, drop_probability=drop_probability,
                                                        dilation_rate=(2, 2))
        bottle_neck_3 = batch_conv_relu_dropout_dilated(input_tensor=bottle_neck_2, filters=1024, name='bottle_neck_3',
                                                        is_training=is_training, drop_probability=drop_probability,
                                                        dilation_rate=(3, 3))
        """ tu5 14x14"""
        up_5 = upscale_concat_batch_conv_relu(input_tensor=bottle_neck_3, concate_tensor=td5,
                                              filters=512, name='up_5', is_training=is_training)
        """ tu4 28x28"""
        up_4 = upscale_concat_batch_conv_relu(input_tensor=up_5, concate_tensor=td4,
                                              filters=512, name='up_4', is_training=is_training)
        """ tu3 56x56"""
        up_3 = upscale_concat_batch_conv_relu(input_tensor=up_4, concate_tensor=td3,
                                              filters=256, name='up_3', is_training=is_training)
        """ tu2 112x112"""
        up_2 = upscale_concat_batch_conv_relu(input_tensor=up_3, concate_tensor=input_tensor,
                                              filters=256, name='up_2', is_training=is_training)
    return up_2


def ae_multi_stack_lite(input_tensor, name, drop_probability, is_training):
    with tf.variable_scope(name):
        """ td3 56x56"""
        td3 = pooling_batch_conv_relu(input_tensor=input_tensor, filters=256, name='td3', is_training=is_training)
        """ td4 28x28"""
        td4 = pooling_batch_conv_relu(input_tensor=td3, filters=512, name='td4', is_training=is_training)
        """ td5 14x14"""
        td5 = pooling_batch_conv_relu(input_tensor=td4, filters=512, name='td5', is_training=is_training)

        """ bottle_neck 7x7"""
        bottle_neck = pooling_batch_conv_relu(input_tensor=td5, filters=512,
                                              name='bottle_neck', is_training=is_training)
        '''
        pool_5 = pool2d(td5)
        bottle_neck_1 = batch_conv_relu_dropout_dilated(input_tensor=pool_5, filters=1024, name='bottle_neck_1',
                                                        is_training=is_training, drop_probability=drop_probability,
                                                        dilation_rate=(1, 1))
        bottle_neck_2 = batch_conv_relu_dropout_dilated(input_tensor=bottle_neck_1, filters=1024, name='bottle_neck_2',
                                                        is_training=is_training, drop_probability=drop_probability,
                                                        dilation_rate=(2, 2))
        bottle_neck_3 = batch_conv_relu_dropout_dilated(input_tensor=bottle_neck_2, filters=1024, name='bottle_neck_3',
                                                        is_training=is_training, drop_probability=drop_probability,
                                                        dilation_rate=(3, 3))
        '''
        """ tu5 14x14"""
        up_5 = upscale_concat_batch_conv_relu(input_tensor=bottle_neck, concate_tensor=td5,
                                              filters=512, name='up_5', is_training=is_training)
        """ tu4 28x28"""
        up_4 = upscale_concat_batch_conv_relu(input_tensor=up_5, concate_tensor=td4,
                                              filters=512, name='up_4', is_training=is_training)
        """ tu3 56x56"""
        up_3 = upscale_concat_batch_conv_relu(input_tensor=up_4, concate_tensor=td3,
                                              filters=256, name='up_3', is_training=is_training)
        """ tu2 112x112"""
        up_2 = upscale_concat_batch_conv_relu(input_tensor=up_3, concate_tensor=input_tensor,
                                              filters=256, name='up_2', is_training=is_training)
    return up_2


def ae_multi(x, flags, drop_probability=0.0, is_training=False):
    x_normal = (x / 127.5) - 1.0
    """ td0 448x448"""
    td0 = conv2d(input_tensor=x_normal, filters=32, activation=tf.nn.relu, name='td0')
    """ td1 224x224"""
    td1 = pooling_batch_conv_relu(input_tensor=td0, filters=64, name='td1', is_training=is_training)
    """ td2 112x112"""
    td2 = pooling_batch_conv_relu(input_tensor=td1, filters=128, name='td2', is_training=is_training)

    """ stacked ae 112x112"""
    stack1 = ae_multi_stack(input_tensor=td2, name='stack1',
                            drop_probability=drop_probability, is_training=is_training)

    stack2 = ae_multi_stack(input_tensor=stack1, name='stack2',
                            drop_probability=drop_probability, is_training=is_training)

    """ output branch 224x224"""
    stack1_up = upscale_concat_batch_conv_relu(input_tensor=stack1, concate_tensor=td1, filters=256,
                                               name='stack1_up', is_training=is_training)
    output_stacked1_sup_sup = tf.layers.conv2d(inputs=stack1_up, filters=3, activation=None,
                                               kernel_size=[1, 1], strides=[1, 1],
                                               padding='same', name='output_stack1_sup_sup')
    output_stacked1_sup = tf.layers.conv2d(inputs=stack1_up, filters=16, activation=None,
                                           kernel_size=[1, 1], strides=[1, 1],
                                           padding='same', name='output_stack1_sup')
    output_stacked1 = tf.layers.conv2d(inputs=stack1_up, filters=92, activation=None,
                                       kernel_size=[1, 1], strides=[1, 1],
                                       padding='same', name='output_stack1')

    stack2_up = upscale_concat_batch_conv_relu(input_tensor=stack2, concate_tensor=td1, filters=256,
                                               name='stack2_up', is_training=is_training)
    output_stacked2_sup_sup = tf.layers.conv2d(inputs=stack2_up, filters=3, activation=None,
                                               kernel_size=[1, 1], strides=[1, 1],
                                               padding='same', name='output_stack2_sup_sup')
    output_stacked2_sup = tf.layers.conv2d(inputs=stack2_up, filters=16, activation=None,
                                           kernel_size=[1, 1], strides=[1, 1],
                                           padding='same', name='output_stack2_sup')
    output_stacked2 = tf.layers.conv2d(inputs=stack2_up, filters=92, activation=None,
                                       kernel_size=[1, 1], strides=[1, 1],
                                       padding='same', name='output_stack2')

    """ bilinear upscale-up 448x448"""
    output_up = tf.image.resize_bilinear(output_stacked2, get_shape(x)[1:3])

    return [output_stacked1_sup_sup, output_stacked1_sup, output_stacked1,
            output_stacked2_sup_sup, output_stacked2_sup, output_stacked2], output_up


def ae_pure(x, flags, drop_probability=0.0, is_training=False):
    x_normal = (x / 127.5) - 1.0
    """ td0 448x448"""
    td0 = conv2d(input_tensor=x_normal, filters=32, activation=tf.nn.relu, name='td0')
    """ td1 224x224"""
    td1 = pooling_batch_conv_relu(input_tensor=td0, filters=64, name='td1', is_training=is_training)
    """ td2 112x112"""
    td2 = pooling_batch_conv_relu(input_tensor=td1, filters=128, name='td2', is_training=is_training)

    """ stacked ae 112x112"""
    stack1 = ae_multi_stack(input_tensor=td2, name='stack1',
                            drop_probability=drop_probability, is_training=is_training)

    # stack2 = ae_multi_stack(input_tensor=stack1, name='stack2',
    #                         drop_probability=drop_probability, is_training=is_training)

    """ output branch 224x224"""
    stack1_up = upscale_concat_batch_conv_relu(input_tensor=stack1, concate_tensor=td1, filters=256,
                                               name='stack1_up', is_training=is_training)
    # output_stacked1_sup_sup = tf.layers.conv2d(inputs=stack1_up, filters=3, activation=None,
    #                                            kernel_size=[1, 1], strides=[1, 1],
    #                                            padding='same', name='output_stack1_sup_sup')
    # output_stacked1_sup = tf.layers.conv2d(inputs=stack1_up, filters=16, activation=None,
    #                                        kernel_size=[1, 1], strides=[1, 1],
    #                                        padding='same', name='output_stack1_sup')
    output_stacked1 = tf.layers.conv2d(inputs=stack1_up, filters=92, activation=None,
                                       kernel_size=[1, 1], strides=[1, 1],
                                       padding='same', name='output_stack1')

    # stack2_up = upscale_concat_batch_conv_relu(input_tensor=stack2, concate_tensor=td1, filters=256,
    #                                            name='stack2_up', is_training=is_training)
    # output_stacked2_sup_sup = tf.layers.conv2d(inputs=stack2_up, filters=3, activation=None,
    #                                            kernel_size=[1, 1], strides=[1, 1],
    #                                            padding='same', name='output_stack2_sup_sup')
    # output_stacked2_sup = tf.layers.conv2d(inputs=stack2_up, filters=16, activation=None,
    #                                        kernel_size=[1, 1], strides=[1, 1],
    #                                        padding='same', name='output_stack2_sup')
    # output_stacked2 = tf.layers.conv2d(inputs=stack2_up, filters=92, activation=None,
    #                                    kernel_size=[1, 1], strides=[1, 1],
    #                                    padding='same', name='output_stack2')

    """ bilinear upscale-up 448x448"""
    output_up = tf.image.resize_bilinear(output_stacked1, get_shape(x)[1:3])

    return [output_stacked1, output_stacked1, output_stacked1], output_up


def ae_pure_stacked(x, flags, drop_probability=0.0, is_training=False):
    x_normal = (x / 127.5) - 1.0
    """ td0 448x448"""
    td0 = conv2d(input_tensor=x_normal, filters=32, activation=tf.nn.relu, name='td0')
    """ td1 224x224"""
    td1 = pooling_batch_conv_relu(input_tensor=td0, filters=64, name='td1', is_training=is_training)
    """ td2 112x112"""
    td2 = pooling_batch_conv_relu(input_tensor=td1, filters=128, name='td2', is_training=is_training)

    """ stacked ae 112x112"""
    stack1 = ae_multi_stack(input_tensor=td2, name='stack1',
                            drop_probability=drop_probability, is_training=is_training)

    # stack2 = ae_multi_stack(input_tensor=stack1, name='stack2',
    #                         drop_probability=drop_probability, is_training=is_training)

    """ output branch 224x224"""
    stack1_up = upscale_concat_batch_conv_relu(input_tensor=stack1, concate_tensor=td1, filters=256,
                                               name='stack1_up', is_training=is_training)
    # output_stacked1_sup_sup = tf.layers.conv2d(inputs=stack1_up, filters=3, activation=None,
    #                                            kernel_size=[1, 1], strides=[1, 1],
    #                                            padding='same', name='output_stack1_sup_sup')
    # output_stacked1_sup = tf.layers.conv2d(inputs=stack1_up, filters=16, activation=None,
    #                                        kernel_size=[1, 1], strides=[1, 1],
    #                                        padding='same', name='output_stack1_sup')
    output_stacked1 = tf.layers.conv2d(inputs=stack1_up, filters=92, activation=None,
                                       kernel_size=[1, 1], strides=[1, 1],
                                       padding='same', name='output_stack1')

    # stack2_up = upscale_concat_batch_conv_relu(input_tensor=stack2, concate_tensor=td1, filters=256,
    #                                            name='stack2_up', is_training=is_training)
    # output_stacked2_sup_sup = tf.layers.conv2d(inputs=stack2_up, filters=3, activation=None,
    #                                            kernel_size=[1, 1], strides=[1, 1],
    #                                            padding='same', name='output_stack2_sup_sup')
    # output_stacked2_sup = tf.layers.conv2d(inputs=stack2_up, filters=16, activation=None,
    #                                        kernel_size=[1, 1], strides=[1, 1],
    #                                        padding='same', name='output_stack2_sup')
    # output_stacked2 = tf.layers.conv2d(inputs=stack2_up, filters=92, activation=None,
    #                                    kernel_size=[1, 1], strides=[1, 1],
    #                                    padding='same', name='output_stack2')

    """ bilinear upscale-up 448x448"""
    output_up = tf.image.resize_bilinear(output_stacked1, get_shape(x)[1:3])

    return [output_stacked1, output_stacked1, output_stacked1], output_up


def ae_pure_multi(x, flags, drop_probability=0.0, is_training=False):
    x_normal = (x / 127.5) - 1.0
    """ td0 448x448"""
    td0 = conv2d(input_tensor=x_normal, filters=32, activation=tf.nn.relu, name='td0')
    """ td1 224x224"""
    td1 = pooling_batch_conv_relu(input_tensor=td0, filters=64, name='td1', is_training=is_training)
    """ td2 112x112"""
    td2 = pooling_batch_conv_relu(input_tensor=td1, filters=128, name='td2', is_training=is_training)

    """ stacked ae 112x112"""
    stack1 = ae_multi_stack(input_tensor=td2, name='stack1',
                            drop_probability=drop_probability, is_training=is_training)

    # stack2 = ae_multi_stack(input_tensor=stack1, name='stack2',
    #                         drop_probability=drop_probability, is_training=is_training)

    """ output branch 224x224"""
    stack1_up = upscale_concat_batch_conv_relu(input_tensor=stack1, concate_tensor=td1, filters=256,
                                               name='stack1_up', is_training=is_training)
    output_stacked1_sup_sup = tf.layers.conv2d(inputs=stack1_up, filters=3, activation=None,
                                               kernel_size=[1, 1], strides=[1, 1],
                                               padding='same', name='output_stack1_sup_sup')
    output_stacked1_sup = tf.layers.conv2d(inputs=stack1_up, filters=16, activation=None,
                                           kernel_size=[1, 1], strides=[1, 1],
                                           padding='same', name='output_stack1_sup')
    output_stacked1 = tf.layers.conv2d(inputs=stack1_up, filters=92, activation=None,
                                       kernel_size=[1, 1], strides=[1, 1],
                                       padding='same', name='output_stack1')

    # stack2_up = upscale_concat_batch_conv_relu(input_tensor=stack2, concate_tensor=td1, filters=256,
    #                                            name='stack2_up', is_training=is_training)
    # output_stacked2_sup_sup = tf.layers.conv2d(inputs=stack2_up, filters=3, activation=None,
    #                                            kernel_size=[1, 1], strides=[1, 1],
    #                                            padding='same', name='output_stack2_sup_sup')
    # output_stacked2_sup = tf.layers.conv2d(inputs=stack2_up, filters=16, activation=None,
    #                                        kernel_size=[1, 1], strides=[1, 1],
    #                                        padding='same', name='output_stack2_sup')
    # output_stacked2 = tf.layers.conv2d(inputs=stack2_up, filters=92, activation=None,
    #                                    kernel_size=[1, 1], strides=[1, 1],
    #                                    padding='same', name='output_stack2')

    """ bilinear upscale-up 448x448"""
    output_up = tf.image.resize_bilinear(output_stacked1, get_shape(x)[1:3])

    return [output_stacked1_sup_sup, output_stacked1_sup, output_stacked1], output_up


def ae_pure_lite(x, flags, drop_probability=0.0, is_training=False):
    x_normal = (x / 127.5) - 1.0
    """ td0 448x448"""
    td0 = conv2d(input_tensor=x_normal, filters=32, activation=tf.nn.relu, name='td0')
    """ td1 224x224"""
    td1 = pooling_batch_conv_relu(input_tensor=td0, filters=64, name='td1', is_training=is_training)
    """ td2 112x112"""
    td2 = pooling_batch_conv_relu(input_tensor=td1, filters=128, name='td2', is_training=is_training)

    """ stacked ae 112x112"""
    stack1 = ae_multi_stack_lite(input_tensor=td2, name='stack1',
                                 drop_probability=drop_probability, is_training=is_training)

    # stack2 = ae_multi_stack(input_tensor=stack1, name='stack2',
    #                         drop_probability=drop_probability, is_training=is_training)

    """ output branch 224x224"""
    stack1_up = upscale_concat_batch_conv_relu(input_tensor=stack1, concate_tensor=td1, filters=256,
                                               name='stack1_up', is_training=is_training)
    # output_stacked1_sup_sup = tf.layers.conv2d(inputs=stack1_up, filters=3, activation=None,
    #                                            kernel_size=[1, 1], strides=[1, 1],
    #                                            padding='same', name='output_stack1_sup_sup')
    # output_stacked1_sup = tf.layers.conv2d(inputs=stack1_up, filters=16, activation=None,
    #                                        kernel_size=[1, 1], strides=[1, 1],
    #                                        padding='same', name='output_stack1_sup')
    output_stacked1 = tf.layers.conv2d(inputs=stack1_up, filters=92, activation=None,
                                       kernel_size=[1, 1], strides=[1, 1],
                                       padding='same', name='output_stack1')

    # stack2_up = upscale_concat_batch_conv_relu(input_tensor=stack2, concate_tensor=td1, filters=256,
    #                                            name='stack2_up', is_training=is_training)
    # output_stacked2_sup_sup = tf.layers.conv2d(inputs=stack2_up, filters=3, activation=None,
    #                                            kernel_size=[1, 1], strides=[1, 1],
    #                                            padding='same', name='output_stack2_sup_sup')
    # output_stacked2_sup = tf.layers.conv2d(inputs=stack2_up, filters=16, activation=None,
    #                                        kernel_size=[1, 1], strides=[1, 1],
    #                                        padding='same', name='output_stack2_sup')
    # output_stacked2 = tf.layers.conv2d(inputs=stack2_up, filters=92, activation=None,
    #                                    kernel_size=[1, 1], strides=[1, 1],
    #                                    padding='same', name='output_stack2')

    """ bilinear upscale-up 448x448"""
    output_up = tf.image.resize_bilinear(output_stacked1, get_shape(x)[1:3])

    return [output_stacked1, output_stacked1, output_stacked1], output_up


def batch_dilated_conv_relu(input_tensor, filters, name, is_training, dilation_rate):
    with tf.variable_scope(name):
        batch = tf.layers.batch_normalization(inputs=input_tensor, training=is_training, name='batch')
        conv = tf.layers.conv2d(inputs=batch, filters=filters, kernel_size=[3, 3], dilation_rate=dilation_rate,
                                kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                strides=[1, 1], padding='same', activation=tf.nn.relu, name='conv')
    return conv


def pooling_batch_conv_relu(input_tensor, filters, name, is_training, k_h=3, k_w=3):
    with tf.variable_scope(name):
        pool = pool2d(input_tensor=input_tensor)
        batch = tf.layers.batch_normalization(inputs=pool, training=is_training, name='batch')
        conv = conv2d(input_tensor=batch, filters=filters, activation=tf.nn.relu, name='conv', k_h=k_h, k_w=k_w)
    return conv


def dilated_se_block(input_tensor, filters, name, is_training, dilation_rate):
    with tf.variable_scope(name):
        """ td3 56x56"""
        td = batch_conv_relu(input_tensor=input_tensor, filters=filters, name='td', is_training=is_training)
        td_se = se(input_tensor=td, name='td_se')
        td_d = batch_dilated_conv_relu(input_tensor=td_se, filters=filters, name='td_d',
                                       is_training=is_training, dilation_rate=dilation_rate)
        td_d_se = se(input_tensor=td_d, name='td_d_se')
    return td_d_se


def ae_lite_dilated_layer(input_tensor, name, drop_probability, is_training):
    with tf.variable_scope(name):
        """ td3 56x56"""
        input_tensor_pool = pool2d(input_tensor)
        td3_d_se = dilated_se_block(input_tensor=input_tensor_pool, filters=256, name='td3_d_se',
                                    is_training=is_training, dilation_rate=(2, 2))
        """ td4 56x56"""
        td4_d_se = dilated_se_block(input_tensor=td3_d_se, filters=512, name='td4_d_se',
                                    is_training=is_training, dilation_rate=(2, 2))
        """ td5 56x56(14x14)"""
        td5_d_se = dilated_se_block(input_tensor=td4_d_se, filters=512, name='td5_d_se',
                                    is_training=is_training, dilation_rate=(2, 2))
        """ td6 56x56(7x7)"""
        td6_d_se = dilated_se_block(input_tensor=td5_d_se, filters=512, name='td6_d_se',
                                    is_training=is_training, dilation_rate=(2, 2))
        """ td7 56x56(3.5x3.5)"""
        td7_d_se = dilated_se_block(input_tensor=td6_d_se, filters=1024, name='td7_d_se',
                                    is_training=is_training, dilation_rate=(2, 2))

        """ tu3 112x112"""
        up_3 = upscale_concat_batch_conv_relu(input_tensor=td7_d_se, concate_tensor=input_tensor,
                                              filters=512, name='up_3', is_training=is_training)
        up_3_se = se(input_tensor=up_3, name='up_3_se')

    return up_3_se


def ae_lite_dilated(x, flags, drop_probability=0.0, is_training=False):
    x_normal = (x / 127.5) - 1.0
    """ td0 448x448"""
    td0 = conv2d(input_tensor=x_normal, filters=32, activation=tf.nn.relu, name='td0')
    td0_se = se(input_tensor=td0, name='td0_se')
    """ td1 224x224"""
    td1 = pooling_batch_conv_relu(input_tensor=td0_se, filters=64, name='td1', is_training=is_training)
    td1_se = se(input_tensor=td1, name='td1_se')
    """ td2 112x112"""
    td2 = pooling_batch_conv_relu(input_tensor=td1_se, filters=128, name='td2', is_training=is_training)
    td2_se = se(input_tensor=td2, name='td2_se')
    """ stacked ae 112x112"""
    stack1 = ae_lite_dilated_layer(input_tensor=td2_se, name='stack1',
                                   drop_probability=drop_probability, is_training=is_training)
    stack1_se = se(input_tensor=stack1, name='stack1_se')

    up_2 = upscale_concat_batch_conv_relu(input_tensor=stack1_se, concate_tensor=td1_se,
                                          filters=512, name='up_2', is_training=is_training)

    # stack2 = ae_lite_dilated_layer(input_tensor=stack1, name='stack2',
    #                                drop_probability=drop_probability, is_training=is_training)

    """ output branch 112x12"""
    # stack1_up = upscale_concat_batch_conv_relu(input_tensor=stack1, concate_tensor=td1, filters=256,
    #                                            name='stack1_up', is_training=is_training)
    output_stacked1 = tf.layers.conv2d(inputs=up_2, filters=92, activation=None,
                                       kernel_size=[1, 1], strides=[1, 1],
                                       padding='same', name='output_stack1')

    # stack2_up = upscale_concat_batch_conv_relu(input_tensor=stack2, concate_tensor=td1, filters=256,
    #                                            name='stack2_up', is_training=is_training)
    # output_stacked2 = tf.layers.conv2d(inputs=stack1 + stack2, filters=92, activation=None,
    #                                    kernel_size=[1, 1], strides=[1, 1],
    #                                    padding='same', name='output_stack2')

    """ bilinear upscale-up 448x448"""
    output_up = tf.image.resize_bilinear(output_stacked1, get_shape(x)[1:3])

    return [output_stacked1], output_up


def batch_dilated_batch_conv_relu(input_tensor, filters, name, is_training, dilation_rate):
    with tf.variable_scope(name):
        batch_d = tf.layers.batch_normalization(inputs=input_tensor, training=is_training, name='batch_dilated')
        dilated = tf.layers.conv2d(inputs=batch_d, filters=filters, kernel_size=[3, 3],  padding='same',
                                   dilation_rate=dilation_rate, activation=tf.nn.relu, name='dilated',
                                   kernel_initializer=tf.contrib.layers.xavier_initializer(), strides=[1, 1])
        batch_c = tf.layers.batch_normalization(inputs=pool, training=is_training, name='batch_conv')
        conv = conv2d(input_tensor=batch, filters=filters, activation=tf.nn.relu, name='conv')
    return conv


def ae_multi_stack_lite_dilated(input_tensor, name, drop_probability, is_training):
    with tf.variable_scope(name):
        """ td3 56x56"""
        td3 = pooling_batch_conv_relu(input_tensor=input_tensor, filters=256, name='td3', is_training=is_training)
        """ td4 28x28"""
        td4 = pooling_batch_conv_relu(input_tensor=td3, filters=512, name='td4', is_training=is_training)
        """ td5 14x14"""
        td5 = pooling_batch_conv_relu(input_tensor=td4, filters=512, name='td5', is_training=is_training)

        """ bottle_neck 7x7"""
        bottle_neck = pooling_batch_conv_relu(input_tensor=td5, filters=512,
                                              name='bottle_neck', is_training=is_training)
        '''
        pool_5 = pool2d(td5)
        bottle_neck_1 = batch_conv_relu_dropout_dilated(input_tensor=pool_5, filters=1024, name='bottle_neck_1',
                                                        is_training=is_training, drop_probability=drop_probability,
                                                        dilation_rate=(1, 1))
        bottle_neck_2 = batch_conv_relu_dropout_dilated(input_tensor=bottle_neck_1, filters=1024, name='bottle_neck_2',
                                                        is_training=is_training, drop_probability=drop_probability,
                                                        dilation_rate=(2, 2))
        bottle_neck_3 = batch_conv_relu_dropout_dilated(input_tensor=bottle_neck_2, filters=1024, name='bottle_neck_3',
                                                        is_training=is_training, drop_probability=drop_probability,
                                                        dilation_rate=(3, 3))
        '''
        """ tu5 14x14"""
        up_5 = upscale_concat_batch_conv_relu(input_tensor=bottle_neck, concate_tensor=td5,
                                              filters=512, name='up_5', is_training=is_training)
        """ tu4 28x28"""
        up_4 = upscale_concat_batch_conv_relu(input_tensor=up_5, concate_tensor=td4,
                                              filters=512, name='up_4', is_training=is_training)
        """ tu3 56x56"""
        up_3 = upscale_concat_batch_conv_relu(input_tensor=up_4, concate_tensor=td3,
                                              filters=256, name='up_3', is_training=is_training)
        """ tu2 112x112"""
        up_2 = upscale_concat_batch_conv_relu(input_tensor=up_3, concate_tensor=input_tensor,
                                              filters=256, name='up_2', is_training=is_training)
    return up_2


def ae_lite_clean(x, flags, drop_probability=0.0, is_training=False):
    x_normal = (x / 127.5) - 1.0
    """ td0 448x448"""
    td0 = conv2d(input_tensor=x_normal, filters=32, activation=tf.nn.relu, name='td0')
    """ td1 224x224"""
    td1 = pooling_batch_conv_relu(input_tensor=td0, filters=64, name='td1', is_training=is_training)
    """ td2 112x112"""
    td2 = pooling_batch_conv_relu(input_tensor=td1, filters=128, name='td2', is_training=is_training)

    """ stacked ae 112x112"""
    stack1 = ae_multi_stack_lite_dilated(input_tensor=td2, name='stack1',
                                         drop_probability=drop_probability, is_training=is_training)

    # stack2 = ae_multi_stack(input_tensor=stack1, name='stack2',
    #                         drop_probability=drop_probability, is_training=is_training)

    """ output branch 224x224"""
    stack1_up = upscale_concat_batch_conv_relu(input_tensor=stack1, concate_tensor=td1, filters=256,
                                               name='stack1_up', is_training=is_training)
    # output_stacked1_sup_sup = tf.layers.conv2d(inputs=stack1_up, filters=3, activation=None,
    #                                            kernel_size=[1, 1], strides=[1, 1],
    #                                            padding='same', name='output_stack1_sup_sup')
    # output_stacked1_sup = tf.layers.conv2d(inputs=stack1_up, filters=16, activation=None,
    #                                        kernel_size=[1, 1], strides=[1, 1],
    #                                        padding='same', name='output_stack1_sup')
    output_stacked1 = tf.layers.conv2d(inputs=stack1_up, filters=92, activation=None,
                                       kernel_size=[1, 1], strides=[1, 1],
                                       padding='same', name='output_stack1')

    # stack2_up = upscale_concat_batch_conv_relu(input_tensor=stack2, concate_tensor=td1, filters=256,
    #                                            name='stack2_up', is_training=is_training)
    # output_stacked2_sup_sup = tf.layers.conv2d(inputs=stack2_up, filters=3, activation=None,
    #                                            kernel_size=[1, 1], strides=[1, 1],
    #                                            padding='same', name='output_stack2_sup_sup')
    # output_stacked2_sup = tf.layers.conv2d(inputs=stack2_up, filters=16, activation=None,
    #                                        kernel_size=[1, 1], strides=[1, 1],
    #                                        padding='same', name='output_stack2_sup')
    # output_stacked2 = tf.layers.conv2d(inputs=stack2_up, filters=92, activation=None,
    #                                    kernel_size=[1, 1], strides=[1, 1],
    #                                    padding='same', name='output_stack2')

    """ bilinear upscale-up 448x448"""
    output_up = tf.image.resize_bilinear(output_stacked1, get_shape(x)[1:3])

    return [output_stacked1], output_up


def ae_lite_stack_layer(input_tensor, name, drop_probability, is_training):
    with tf.variable_scope(name):
        """ td3 56x56"""
        td3 = pooling_batch_conv_relu(input_tensor=input_tensor, filters=256, name='td3', is_training=is_training)
        """ td4 28x28"""
        td4 = pooling_batch_conv_relu(input_tensor=td3, filters=512, name='td4', is_training=is_training)
        """ td5 14x14"""
        td5 = pooling_batch_conv_relu(input_tensor=td4, filters=512, name='td5', is_training=is_training)

        """ bottle_neck 7x7"""
        bottle_neck = pooling_batch_conv_relu(input_tensor=td5, filters=512,
                                              name='bottle_neck', is_training=is_training)
        '''
        pool_5 = pool2d(td5)
        bottle_neck_1 = batch_conv_relu_dropout_dilated(input_tensor=pool_5, filters=1024, name='bottle_neck_1',
                                                        is_training=is_training, drop_probability=drop_probability,
                                                        dilation_rate=(1, 1))
        bottle_neck_2 = batch_conv_relu_dropout_dilated(input_tensor=bottle_neck_1, filters=1024, name='bottle_neck_2',
                                                        is_training=is_training, drop_probability=drop_probability,
                                                        dilation_rate=(2, 2))
        bottle_neck_3 = batch_conv_relu_dropout_dilated(input_tensor=bottle_neck_2, filters=1024, name='bottle_neck_3',
                                                        is_training=is_training, drop_probability=drop_probability,
                                                        dilation_rate=(3, 3))
        '''
        """ tu5 14x14"""
        up_5 = upscale_concat_batch_conv_relu(input_tensor=bottle_neck, concate_tensor=td5,
                                              filters=512, name='up_5', is_training=is_training)
        """ tu4 28x28"""
        up_4 = upscale_concat_batch_conv_relu(input_tensor=up_5, concate_tensor=td4,
                                              filters=512, name='up_4', is_training=is_training)
        """ tu3 56x56"""
        up_3 = upscale_concat_batch_conv_relu(input_tensor=up_4, concate_tensor=td3,
                                              filters=256, name='up_3', is_training=is_training)
        """ tu2 112x112"""
        up_2 = upscale_concat_batch_conv_relu(input_tensor=up_3, concate_tensor=input_tensor,
                                              filters=256, name='up_2', is_training=is_training)
    return up_2


def ae_lite_stack(x, flags, drop_probability=0.0, is_training=False):
    x_normal = (x / 127.5) - 1.0
    """ td0 448x448"""
    td0 = conv2d(input_tensor=x_normal, filters=32, activation=tf.nn.relu, name='td0')
    """ td1 224x224"""
    td1 = pooling_batch_conv_relu(input_tensor=td0, filters=64, name='td1', is_training=is_training)
    """ td2 112x112"""
    td2 = pooling_batch_conv_relu(input_tensor=td1, filters=128, name='td2', is_training=is_training)

    """ stacked ae 112x112"""
    stack1 = ae_lite_stack_layer(input_tensor=td2, name='stack1',
                                 drop_probability=drop_probability, is_training=is_training)

    stack2 = ae_lite_stack_layer(input_tensor=stack1, name='stack2',
                                 drop_probability=drop_probability, is_training=is_training)

    """ output branch 224x224"""
    stack1_up = upscale_concat_batch_conv_relu(input_tensor=stack1, concate_tensor=td1, filters=256,
                                               name='stack1_up', is_training=is_training)
    output_stacked1 = tf.layers.conv2d(inputs=stack1_up, filters=92, activation=None,
                                       kernel_size=[1, 1], strides=[1, 1],
                                       padding='same', name='output_stack1')

    stack2_up = upscale_concat_batch_conv_relu(input_tensor=stack2, concate_tensor=td1, filters=256,
                                               name='stack2_up', is_training=is_training)
    output_stacked2 = tf.layers.conv2d(inputs=stack1_up + stack2_up, filters=92, activation=None,
                                       kernel_size=[1, 1], strides=[1, 1],
                                       padding='same', name='output_stack2')

    """ bilinear upscale-up 448x448"""
    output_up = tf.image.resize_bilinear(output_stacked2, get_shape(x)[1:3])

    return [output_stacked1, output_stacked2], output_up


def batch_conv_relu(input_tensor, filters, is_training, name, k_h=3, k_w=3, d_h=1, d_w=1):
    with tf.variable_scope(name):
        batch = tf.layers.batch_normalization(inputs=input_tensor, training=is_training, name='batch')
        conv = conv2d(input_tensor=batch, filters=filters, activation=tf.nn.relu, name='conv',
                      k_h=k_h, k_w=k_w, d_h=d_h, d_w=d_w)
    return conv



def ae_lite_full(x, flags, drop_probability=0.0, is_training=False):
    x_normal = (x / 127.5) - 1.0
    """ td0 448x448"""
    td0 = conv2d(input_tensor=x_normal, filters=32, activation=tf.nn.relu, name='td0')
    """ td1 224x224"""
    td1 = pooling_batch_conv_relu(input_tensor=td0, filters=64, name='td1', is_training=is_training)
    """ td2 112x112"""
    td2 = pooling_batch_conv_relu(input_tensor=td1, filters=128, name='td2', is_training=is_training)
    """ td3 56x56"""
    td3 = pooling_batch_conv_relu(input_tensor=td2, filters=256, name='td3', is_training=is_training)
    """ td4 28x28"""
    td4 = pooling_batch_conv_relu(input_tensor=td3, filters=512, name='td4', is_training=is_training)
    """ td5 14x14"""
    td5 = pooling_batch_conv_relu(input_tensor=td4, filters=512, name='td5', is_training=is_training)

    """ bottle_neck 7x7"""
    # bottle_neck = pooling_batch_conv_relu(input_tensor=td5, filters=1024,
    #                                       name='bottle_neck', is_training=is_training)
    pool5 = pool2d(input_tensor=td5)
    bottle_neck1 = batch_dilated_conv_relu(input_tensor=pool5, filters=1024, name='bottle_neck1',
                                           is_training=is_training, dilation_rate=(1, 1))
    bottle_neck2 = batch_dilated_conv_relu(input_tensor=bottle_neck1, filters=1024, name='bottle_neck2',
                                           is_training=is_training, dilation_rate=(2, 2))
    bottle_neck3 = batch_dilated_conv_relu(input_tensor=bottle_neck2, filters=1024, name='bottle_neck3',
                                           is_training=is_training, dilation_rate=(3, 3))
    """ tu5 14x14"""
    up_5 = upscale_concat_batch_conv_relu(input_tensor=bottle_neck3, concate_tensor=td5,
                                          filters=512, name='up_5', is_training=is_training)
    """ tu4 28x28"""
    up_4 = upscale_concat_batch_conv_relu(input_tensor=up_5, concate_tensor=td4,
                                          filters=512, name='up_4', is_training=is_training)
    """ tu3 56x56"""
    up_3 = upscale_concat_batch_conv_relu(input_tensor=up_4, concate_tensor=td3,
                                          filters=256, name='up_3', is_training=is_training)
    """ tu2 112x112"""
    up_2 = upscale_concat_batch_conv_relu(input_tensor=up_3, concate_tensor=td2,
                                          filters=256, name='up_2', is_training=is_training)
    """ tu2 224x224"""
    # up_1 = upscale_concat_batch_conv_relu(input_tensor=up_2, concate_tensor=td1,
    #                                       filters=128, name='up_1', is_training=is_training)
    # up_1 = upscale(up_2)
    """ output 448x448"""
    feature_up = upscale(up_2, s_w=4, s_h=4)
    feature_concat = tf.concat([feature_up, td0], 3)
    batch = tf.layers.batch_normalization(inputs=feature_concat, training=is_training, name='batch')
    output = conv2d(input_tensor=batch, filters=flags.num_of_class, activation=None, name='output')

    return [output], output


def ae_lite_se(x, flags, drop_probability=0.0, is_training=False):
    x_normal = (x / 127.5) - 1.0
    """ td0 448x448"""
    td0 = conv2d(input_tensor=x_normal, filters=32, activation=tf.nn.relu, name='td0')
    """ td1 224x224"""
    td1 = pooling_batch_conv_relu(input_tensor=td0, filters=64, name='td1', is_training=is_training)
    td1_se = se(input_tensor=td1, name='td1_se')
    """ td2 112x112"""
    td2 = pooling_batch_conv_relu(input_tensor=td1_se, filters=128, name='td2', is_training=is_training)
    td2_se = se(input_tensor=td2, name='td2_se')
    """ td3 56x56"""
    td3 = pooling_batch_conv_relu(input_tensor=td2_se, filters=256, name='td3', is_training=is_training)
    td3_se = se(input_tensor=td3, name='td3_se')
    """ td4 28x28"""
    td4 = pooling_batch_conv_relu(input_tensor=td3_se, filters=512, name='td4', is_training=is_training)
    td4_se = se(input_tensor=td4, name='td4_se')
    """ td5 14x14"""
    td5 = pooling_batch_conv_relu(input_tensor=td4_se, filters=512, name='td5', is_training=is_training)
    td5_se = se(input_tensor=td5, name='td5_se')

    """ bottle_neck 7x7"""
    # bottle_neck = pooling_batch_conv_relu(input_tensor=td5, filters=1024,
    #                                       name='bottle_neck', is_training=is_training)
    pool5 = pool2d(input_tensor=td5_se)
    bottle_neck1 = batch_dilated_conv_relu(input_tensor=pool5, filters=1024, name='bottle_neck1',
                                           is_training=is_training, dilation_rate=(1, 1))
    bottle_neck1_se = se(input_tensor=bottle_neck1, name='bottle_neck1_se')

    bottle_neck2 = batch_dilated_conv_relu(input_tensor=bottle_neck1_se, filters=1024, name='bottle_neck2',
                                           is_training=is_training, dilation_rate=(2, 2))
    bottle_neck2_se = se(input_tensor=bottle_neck2, name='bottle_neck2_se')

    bottle_neck3 = batch_dilated_conv_relu(input_tensor=bottle_neck2_se, filters=1024, name='bottle_neck3',
                                           is_training=is_training, dilation_rate=(3, 3))
    bottle_neck3_se = se(input_tensor=bottle_neck3, name='bottle_neck3_se')

    """ tu5 14x14"""
    up_5 = upscale_concat_batch_conv_relu(input_tensor=bottle_neck3_se, concate_tensor=td5_se,
                                          filters=512, name='up_5', is_training=is_training)
    up5_se = se(input_tensor=up_5, name='up5_se')
    """ tu4 28x28"""
    up_4 = upscale_concat_batch_conv_relu(input_tensor=up5_se, concate_tensor=td4_se,
                                          filters=512, name='up_4', is_training=is_training)
    up4_se = se(input_tensor=up_4, name='up4_se')
    """ tu3 56x56"""
    up_3 = upscale_concat_batch_conv_relu(input_tensor=up4_se, concate_tensor=td3_se,
                                          filters=256, name='up_3', is_training=is_training)
    up3_se = se(input_tensor=up_3, name='up3_se')
    """ tu2 112x112"""
    up_2 = upscale_concat_batch_conv_relu(input_tensor=up3_se, concate_tensor=td2_se,
                                          filters=256, name='up_2', is_training=is_training)
    up2_se = se(input_tensor=up_2, name='up2_se')
    """ output 224x224"""
    up_1 = upscale_concat_batch_conv_relu(input_tensor=up2_se, concate_tensor=td1_se,
                                          filters=256, name='up_1', is_training=is_training)

    output = tf.layers.conv2d(inputs=up_1, filters=flags.num_of_class, activation=None,
                              kernel_size=[1, 1], strides=[1, 1], padding='same', name='output')

    """ bilinear upscale-up 448x448"""
    output_up = tf.image.resize_bilinear(output, get_shape(x)[1:3])

    return [output], output_up


def relu_batch_conv(input_tensor, filters, is_training, name, k_h=3, k_w=3, d_h=1, d_w=1):
    with tf.variable_scope(name):
        relu = tf.nn.relu(features=input_tensor, name='relu')
        batch = tf.layers.batch_normalization(inputs=relu, training=is_training, name='batch')
        conv = conv2d(input_tensor=batch, filters=filters, activation=None, name='conv',
                      k_h=k_h, k_w=k_w, d_h=d_h, d_w=d_w)
    return conv


def se(input_tensor, name):
    with tf.variable_scope(name):
        shape = get_shape(input_tensor)
        # global_pooling = tf.layers.average_pooling2d(inputs=input_tensor, pool_size=(shape[1], shape[2]),
        #                                              strides=(1, 1), padding='valid')
        global_pooling = tf.reduce_mean(input_tensor, [1, 2])
        # global_pooling_flatten = tf.reshape(global_pooling, [-1])
        # global_pooling_flatten = tf.contrib.layers.flatten(global_pooling)
        fc_1 = tf.layers.dense(inputs=global_pooling, units=(shape[3] // 16), activation=tf.nn.relu,
                               kernel_initializer=tf.contrib.layers.xavier_initializer(), name='fc1')
        fc_2 = tf.layers.dense(inputs=fc_1, units=(shape[3]), activation=tf.sigmoid,
                               kernel_initializer=tf.contrib.layers.xavier_initializer(), name='fc2')
        fc_2 = tf.expand_dims(fc_2, axis=1)
        fc_2 = tf.expand_dims(fc_2, axis=2)
        x_scale = input_tensor * fc_2

    return x_scale


def residual_block(input_tensor, filters, is_training, name, k_h=3, k_w=3, d_h=1, d_w=1, activation=None):
    with tf.variable_scope(name):
        relu1 = tf.nn.relu(features=input_tensor, name='relu1')
        batch1 = tf.layers.batch_normalization(inputs=relu1, training=is_training, name='batch1')
        conv1 = conv2d(input_tensor=batch1, filters=filters, activation=activation, name='conv1',
                       k_h=k_h, k_w=k_w, d_h=d_h, d_w=d_w)
        se1 = se(input_tensor=conv1, name='se1')

        relu2 = tf.nn.relu(features=se1, name='relu2')
        batch2 = tf.layers.batch_normalization(inputs=relu2, training=is_training, name='batch2')
        conv2 = conv2d(input_tensor=batch1, filters=batch2, activation=activation, name='conv2',
                       k_h=k_h, k_w=k_w, d_h=d_h, d_w=d_w)
        se2 = se(input_tensor=conv2, name='se2')
    return input_tensor + se2


def conv_se(input_tensor, filters, name, k_h=3, k_w=3, d_h=1, d_w=1, activation=None):
    with tf.variable_scope(name):
        conv = conv2d(input_tensor=input_tensor, filters=filters, activation=activation, name='conv',
                      k_h=k_h, k_w=k_w, d_h=d_h, d_w=d_w)
        feature = se(input_tensor=conv, name='se')
    return feature


def relu_batch_dilated_conv_se(input_tensor, filters, name, is_training, dilation_rate):
    with tf.variable_scope(name):
        relu = tf.nn.relu(features=input_tensor, name='relu')
        batch = tf.layers.batch_normalization(inputs=relu, training=is_training, name='batch')
        conv = tf.layers.conv2d(inputs=batch, filters=filters, kernel_size=[3, 3], dilation_rate=dilation_rate,
                                kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                strides=[1, 1], padding='same', activation=tf.nn.relu, name='conv')
        feature = se(input_tensor=conv, name='se')
    return feature


def relu_batch_conv_se(input_tensor, filters, is_training, name, k_h=3, k_w=3, d_h=1, d_w=1, activation=None):
    with tf.variable_scope(name):
        relu = tf.nn.relu(features=input_tensor, name='relu')
        batch = tf.layers.batch_normalization(inputs=relu, training=is_training, name='batch')
        feature = conv_se(input_tensor=batch, filters=filters, activation=activation, name='conv_se',
                          k_h=k_h, k_w=k_w, d_h=d_h, d_w=d_w)
    return feature


def residual_block2(input_tensor, filters, is_training, name, k_h=3, k_w=3, d_h=1, d_w=1, activation=None):
    with tf.variable_scope(name):
        feature1 = relu_batch_conv_se(input_tensor=input_tensor, filters=filters, is_training=is_training,
                                      name='feature1', k_h=1, k_w=1, d_h=d_h, d_w=d_w, activation=None)
        feature2 = relu_batch_conv_se(input_tensor=feature1, filters=filters, is_training=is_training,
                                      name='feature2', k_h=3, k_w=3, d_h=d_h, d_w=d_w, activation=None)
        feature3 = relu_batch_conv_se(input_tensor=feature2, filters=(2 * filters), is_training=is_training,
                                      name='feature4', k_h=1, k_w=1, d_h=d_h, d_w=d_w, activation=None)
    return input_tensor + feature3


def residual_module(input_tensor, filters, is_training, name, layers_num):
    with tf.variable_scope(name):
        input_residual = relu_batch_conv_se(input_tensor=input_tensor, filters=filters, activation=None,
                                            name='residual_0', is_training=is_training, d_h=2, d_w=2)
        for i in range(layers_num):
            input_residual = residual_block2(input_tensor=input_residual, filters=filters // 2,
                                             is_training=is_training, name='residual_{:d}'.format(i))
    return input_residual


def upscale_concat_relu_batch_conv_se(input_tensor, concate_tensor, filters, is_training, name):
    with tf.variable_scope(name):
        feature_up = upscale(input_tensor)
        feature_concat = tf.concat([feature_up, concate_tensor], 3)
        feature = relu_batch_conv_se(input_tensor=feature_concat, filters=filters, is_training=is_training,
                                     name='feature', activation=None)
    return feature


def class_predict(input_tensor, name, is_training):
    with tf.variable_scope(name):
        relu1 = tf.nn.relu(input_tensor)
        batch1 = tf.layers.batch_normalization(inputs=relu1, training=is_training, name='batch1')
        predict_sup_sup = tf.layers.conv2d(inputs=batch1, filters=3, activation=None, kernel_size=[1, 1],
                                           strides=[1, 1], padding='same', name='predict_sup_sup')

        relu2 = tf.nn.relu(tf.concat([input_tensor, predict_sup_sup], 3))
        batch2 = tf.layers.batch_normalization(inputs=relu2, training=is_training, name='batch2')
        predict_sup = tf.layers.conv2d(inputs=batch2, filters=16, activation=None, kernel_size=[1, 1],
                                       strides=[1, 1],  padding='same', name='predict_sup')

        relu3 = tf.nn.relu(tf.concat([input_tensor, predict_sup_sup, predict_sup], 3))
        batch3 = tf.layers.batch_normalization(inputs=relu3, training=is_training, name='batch3')
        predict = tf.layers.conv2d(inputs=batch3, filters=92, activation=None, kernel_size=[1, 1],
                                   strides=[1, 1], padding='same', name='predict')

    return [predict_sup_sup, predict_sup, predict]


def ae_lite_se_final(x, flags, drop_probability=0.0, is_training=False):
    x_normal = (x / 127.5) - 1.0
    """ td0 448x448"""
    # td0 = conv2d(input_tensor=x_normal, filters=32, activation=tf.nn.relu, name='td0')
    """ td1 224x224"""
    td1 = conv_se(input_tensor=x_normal, filters=32, activation=None, name='td1', k_h=7, k_w=7, d_h=2, d_w=2)
    """ td2 112x112"""
    td2 = relu_batch_conv_se(input_tensor=td1, filters=64, activation=None, name='td2', is_training=is_training,
                             k_h=7, k_w=7, d_h=2, d_w=2)
    # ------------------------------------------------------------------------------------------------
    """ td3 56x56"""
    td3 = residual_module(input_tensor=td2, filters=128, is_training=is_training, name='td3', layers_num=3)
    """ td4 28x28"""
    td4 = residual_module(input_tensor=td3, filters=256, is_training=is_training, name='td4', layers_num=4)
    """ td5 14x14"""
    td5 = residual_module(input_tensor=td4, filters=512, is_training=is_training, name='td5', layers_num=6)

    """ bottle_neck 7x7"""
    bottle_neck_in = relu_batch_conv_se(input_tensor=td5, filters=1024, activation=None, name='bottle_neck_in',
                                        is_training=is_training, d_h=2, d_w=2)
    bottle_neck1 = relu_batch_dilated_conv_se(input_tensor=bottle_neck_in, filters=1024, name='bottle_neck1',
                                              is_training=is_training, dilation_rate=(2, 2))
    bottle_neck2 = relu_batch_dilated_conv_se(input_tensor=bottle_neck1, filters=1024, name='bottle_neck3',
                                              is_training=is_training, dilation_rate=(3, 3))
    bottle_neck = tf.add(x=bottle_neck2, y=bottle_neck_in, name='bottle_neck')

    """ tu5 14x14"""
    up_5 = upscale_concat_relu_batch_conv_se(input_tensor=bottle_neck, concate_tensor=td5,
                                             filters=512, name='up_5', is_training=is_training)
    """ tu4 28x28"""
    up_4 = upscale_concat_relu_batch_conv_se(input_tensor=up_5, concate_tensor=td4,
                                             filters=512, name='up_4', is_training=is_training)
    """ tu3 56x56"""
    up_3 = upscale_concat_relu_batch_conv_se(input_tensor=up_4, concate_tensor=td3,
                                             filters=256, name='up_3', is_training=is_training)
    # prediction_56 = class_predict(input_tensor=up_3, name='prediction_56')
    """ tu2 112x112"""
    up_2 = upscale_concat_relu_batch_conv_se(input_tensor=up_3, concate_tensor=td2,
                                             filters=256, name='up_2', is_training=is_training)
    # prediction_112 = class_predict(input_tensor=up_2, name='prediction_112')
    """ output 224x224"""
    up_1 = upscale_concat_relu_batch_conv_se(input_tensor=up_2, concate_tensor=td1,
                                             filters=256, name='up_1', is_training=is_training)

    prediction_224 = class_predict(name='prediction_224', is_training=is_training,
                                   input_tensor=(up_1 + upscale_blinear(up_2, 2, 2) + upscale_blinear(up_3, 4, 4)))

    # output_sup_sup = upscale_blinear(prediction_56[0], 4, 4) + upscale_blinear(prediction_112[0], 2, 2) + prediction_224[0]
    # output_sup = upscale_blinear(prediction_56[1], 4, 4) + upscale_blinear(prediction_112[1], 2, 2) + prediction_224[1]
    # output = upscale_blinear(prediction_56[2], 4, 4) + upscale_blinear(prediction_112[2], 2, 2) + prediction_224[2]
    output_sup_sup = prediction_224[0]
    output_sup = prediction_224[1]
    output = prediction_224[2]
    """ bilinear upscale-up 448x448"""
    output_up = tf.image.resize_bilinear(output, get_shape(x)[1:3])

    return [output_sup_sup, output_sup, output], output_up


def stacked2_nearest(x, flags, drop_probability=0.0, is_training=False):
    """ Original size 448x448"""
    """ bilinear scale-down 224x224"""
    x_down = tf.image.resize_bilinear(x, (get_shape(x)[1] // 2, get_shape(x)[2] // 2))
    x_down = (x_down / 127.5) - 1.0
    """ td1 224x224"""
    feature1 = dense_block_down(input_tensor=x_down, num_layers=4, growth_rate=8, filters=32,
                                name='feature_td1', is_training=is_training, drop_probability=drop_probability)
    # ------------------------------------------------------------------------------------------
    # Use conv instead of resize to learn the useful features
    feature1_pool = pool2d(input_tensor=feature1)
    feature2 = dense_block_down(input_tensor=feature1_pool, num_layers=4, growth_rate=16, filters=64,
                                name='feature_td2', is_training=is_training, drop_probability=drop_probability)
    """ td3 56x56"""
    feature2_pool = pool2d(feature2)
    stack_feature1 = ae_stack_layer(input_tensor=feature2_pool, name='stack1',
                                    drop_probability=drop_probability, is_training=is_training)

    stack_feature2 = ae_stack_layer(input_tensor=stack_feature1, name='stack2',
                                    drop_probability=drop_probability, is_training=is_training)

    """ output 56x56"""
    output_stacked1 = tf.layers.conv2d(inputs=stack_feature1, filters=flags.num_of_class, activation=None,
                                       kernel_size=[1, 1], strides=[1, 1], padding='same', name='output_stacked1')

    output_stacked2 = tf.layers.conv2d(inputs=stack_feature2, filters=flags.num_of_class, activation=None,
                                       kernel_size=[1, 1], strides=[1, 1], padding='same', name='output_stacked2')

    """ bilinear upscale-up 512x512"""
    output_up = tf.image.resize_bilinear(output_stacked2, get_shape(x)[1:3])

    return [output_stacked1, output_stacked2], output_up


def stacked2_stride(x, flags, drop_probability=0.0, is_training=False):
    """ Original size 448x448"""
    """ bilinear scale-down 224x224"""
    x_down = tf.image.resize_bilinear(x, (get_shape(x)[1] // 2, get_shape(x)[2] // 2))
    x_down = (x_down / 127.5) - 1.0
    """ td1 224x224"""
    feature1 = dense_block_down(input_tensor=x_down, num_layers=4, growth_rate=8, filters=32,
                                name='feature_td1', is_training=is_training, drop_probability=drop_probability)
    # ------------------------------------------------------------------------------------------
    # Use conv instead of resize to learn the useful features
    feature1_pool = pool2d(input_tensor=feature1)
    feature2 = dense_block_down(input_tensor=feature1_pool, num_layers=4, growth_rate=16, filters=64,
                                name='feature_td2', is_training=is_training, drop_probability=drop_probability)
    """ td3 56x56"""
    feature2_pool = pool2d(feature2)
    stack_feature1 = ae_stack_layer_stride(input_tensor=feature2_pool, name='stack1',
                                           drop_probability=drop_probability, is_training=is_training)

    stack_feature2 = ae_stack_layer_stride(input_tensor=stack_feature1, name='stack2',
                                           drop_probability=drop_probability, is_training=is_training)

    """ output 56x56"""
    output_stacked1 = tf.layers.conv2d(inputs=stack_feature1, filters=flags.num_of_class, activation=None,
                                       kernel_size=[1, 1], strides=[1, 1], padding='same', name='output_stacked1')

    output_stacked2 = tf.layers.conv2d(inputs=stack_feature2, filters=flags.num_of_class, activation=None,
                                       kernel_size=[1, 1], strides=[1, 1], padding='same', name='output_stacked2')

    """ bilinear upscale-up 512x512"""
    output_up = tf.image.resize_bilinear(output_stacked2, get_shape(x)[1:3])

    return [output_stacked1, output_stacked2], output_up


def stacked2_stride_dense(x, flags, drop_probability=0.0, is_training=False):
    """ Original size 448x448"""
    """ bilinear scale-down 224x224"""
    x_down = tf.image.resize_bilinear(x, (get_shape(x)[1] // 2, get_shape(x)[2] // 2))
    x_down = (x_down / 127.5) - 1.0
    """ td1 224x224"""
    feature1 = dense_block_down(input_tensor=x_down, num_layers=4, growth_rate=8, filters=32,
                                name='feature_td1', is_training=is_training, drop_probability=drop_probability)
    # ------------------------------------------------------------------------------------------
    # Use conv instead of resize to learn the useful features
    feature1_pool = pool2d(input_tensor=feature1)
    feature2 = dense_block_down(input_tensor=feature1_pool, num_layers=4, growth_rate=16, filters=64,
                                name='feature_td2', is_training=is_training, drop_probability=drop_probability)
    """ td3 56x56"""
    feature2_pool = pool2d(feature2)
    stack_feature1 = stacked_layer_stride(input_tensor=feature2_pool, name='stack1',
                                          drop_probability=drop_probability, is_training=is_training)

    stack_feature2 = stacked_layer_stride(input_tensor=stack_feature1, name='stack2',
                                          drop_probability=drop_probability, is_training=is_training)

    """ output 56x56"""
    output_stacked1 = tf.layers.conv2d(inputs=stack_feature1, filters=flags.num_of_class, activation=None,
                                       kernel_size=[1, 1], strides=[1, 1], padding='same', name='output_stacked1')

    output_stacked2 = tf.layers.conv2d(inputs=stack_feature2, filters=flags.num_of_class, activation=None,
                                       kernel_size=[1, 1], strides=[1, 1], padding='same', name='output_stacked2')

    """ bilinear upscale-up 512x512"""
    output_up = tf.image.resize_bilinear(output_stacked2, get_shape(x)[1:3])

    return [output_stacked1, output_stacked2], output_up


def dense_ae_stacked_stride(x, flags, drop_probability=0.0, is_training=False):
    """ Original size 448x448"""
    """ bilinear scale-down 224x224"""
    x_down = tf.image.resize_bilinear(x, (get_shape(x)[1] // 2, get_shape(x)[2] // 2))
    x_down = (x_down / 127.5) - 1.0
    """ td1 224x224"""
    feature1 = dense_block_down(input_tensor=x_down, num_layers=4, growth_rate=8, filters=32,
                                name='feature_td1', is_training=is_training, drop_probability=drop_probability)
    # ------------------------------------------------------------------------------------------
    # Use conv instead of resize to learn the useful features
    feature1_pool = pool2d(input_tensor=feature1)
    feature2 = dense_block_down(input_tensor=feature1_pool, num_layers=4, growth_rate=16, filters=64,
                                name='feature_td2', is_training=is_training, drop_probability=drop_probability)
    """ td3 56x56"""
    feature2_pool = pool2d(feature2)
    stack_feature1 = ae_stack_layer_stride(input_tensor=feature2_pool, name='stack1',
                                           drop_probability=drop_probability, is_training=is_training)

    # stack_feature1_concat = tf.concat([feature2_pool, stack_feature1], 3)
    stack_feature2 = ae_stack_layer_stride(input_tensor=stack_feature1, name='stack2',
                                           drop_probability=drop_probability, is_training=is_training)

    # stack_feature2_concat = tf.concat([feature2_pool, stack_feature2], 3)
    # stack_feature3 = ae_stack_layer(input_tensor=stack_feature2, name='stack3',
    #                                 drop_probability=drop_probability, is_training=is_training)

    """ output 56x56"""
    output_stacked1 = tf.layers.conv2d(inputs=stack_feature1, filters=flags.num_of_class, activation=None,
                                       kernel_size=[1, 1], strides=[1, 1], padding='same', name='output_stacked1')

    # output_stacked1_concat = tf.concat([stack_feature2, output_stacked1], 3)
    output_stacked2 = tf.layers.conv2d(inputs=stack_feature2, filters=flags.num_of_class, activation=None,
                                       kernel_size=[1, 1], strides=[1, 1], padding='same', name='output_stacked2')

    # output_stacked2_concat = tf.concat([stack_feature3, output_stacked2], 3)
    # output_stacked3 = tf.layers.conv2d(inputs=stack_feature3, filters=flags.num_of_class, activation=None,
    #                                    kernel_size=[1, 1], strides=[1, 1], padding='same', name='output_stacked3')

    """ bilinear upscale-up 512x512"""
    # output_up = tf.image.resize_nearest_neighbor(output_stacked2, get_shape(x)[1:3])
    output_up = tf.image.resize_bilinear(output_stacked2, get_shape(x)[1:3])

    return [output_stacked1, output_stacked2],  output_up


def dense_ae_stacked(x, flags, drop_probability=0.0, is_training=False):
    """ Original size 448x448"""
    """ bilinear scale-down 224x224"""
    x_down = tf.image.resize_bilinear(x, (get_shape(x)[1] // 2, get_shape(x)[2] // 2))
    x_down = (x_down / 127.5) - 1.0
    """ td1 224x224"""
    feature1 = dense_block_down(input_tensor=x_down, num_layers=4, growth_rate=8, filters=32,
                                name='feature_td1', is_training=is_training, drop_probability=drop_probability)
    # ------------------------------------------------------------------------------------------
    # Use conv instead of resize to learn the useful features
    feature1_pool = pool2d(input_tensor=feature1)
    feature2 = dense_block_down(input_tensor=feature1_pool, num_layers=4, growth_rate=16, filters=64,
                                name='feature_td2', is_training=is_training, drop_probability=drop_probability)
    """ td3 56x56"""
    feature2_pool = pool2d(feature2)
    stack_feature1 = ae_stack_layer(input_tensor=feature2_pool, name='stack1',
                                    drop_probability=drop_probability, is_training=is_training)

    stack_feature1_concat = tf.concat([feature2_pool, stack_feature1], 3)
    stack_feature2 = ae_stack_layer(input_tensor=stack_feature1_concat, name='stack2',
                                    drop_probability=drop_probability, is_training=is_training)

    stack_feature2_concat = tf.concat([feature2_pool, stack_feature2], 3)
    stack_feature3 = ae_stack_layer(input_tensor=stack_feature2_concat, name='stack3',
                                    drop_probability=drop_probability, is_training=is_training)

    """ output 56x56"""
    output_stacked1 = tf.layers.conv2d(inputs=stack_feature1, filters=flags.num_of_class, activation=None,
                                       kernel_size=[1, 1], strides=[1, 1], padding='same', name='output_stacked1')

    output_stacked1_concat = tf.concat([stack_feature2, output_stacked1], 3)
    output_stacked2 = tf.layers.conv2d(inputs=output_stacked1_concat, filters=flags.num_of_class, activation=None,
                                       kernel_size=[1, 1], strides=[1, 1], padding='same', name='output_stacked2')

    output_stacked2_concat = tf.concat([stack_feature3, output_stacked2], 3)
    output_stacked3 = tf.layers.conv2d(inputs=output_stacked2_concat, filters=flags.num_of_class, activation=None,
                                       kernel_size=[1, 1], strides=[1, 1], padding='same', name='output_stacked3')

    """ bilinear upscale-up 512x512"""
    # output_up = tf.image.resize_nearest_neighbor(output_stacked2, get_shape(x)[1:3])
    output_up = tf.image.resize_bilinear(output_stacked3, get_shape(x)[1:3])

    return [output_stacked1, output_stacked2, output_stacked3],  output_up


def dense_ae_stacked_wo(x, flags, drop_probability=0.0, is_training=False):
    """ Original size 448x448"""
    """ bilinear scale-down 224x224"""
    x_down = tf.image.resize_bilinear(x, (get_shape(x)[1] // 2, get_shape(x)[2] // 2))
    x_down = (x_down / 127.5) - 1.0
    """ td1 224x224"""
    feature1 = dense_block_down(input_tensor=x_down, num_layers=4, growth_rate=8, filters=32,
                                name='feature_td1', is_training=is_training, drop_probability=drop_probability)
    # ------------------------------------------------------------------------------------------
    # Use conv instead of resize to learn the useful features
    feature1_pool = pool2d(input_tensor=feature1)
    feature2 = dense_block_down(input_tensor=feature1_pool, num_layers=4, growth_rate=16, filters=64,
                                name='feature_td2', is_training=is_training, drop_probability=drop_probability)
    """ td3 56x56"""
    feature2_pool = pool2d(feature2)
    stack_feature1 = ae_stack_layer(input_tensor=feature2_pool, name='stack1',
                                    drop_probability=drop_probability, is_training=is_training)

    # stack_feature1_concat = tf.concat([feature2_pool, stack_feature1], 3)
    stack_feature2 = ae_stack_layer(input_tensor=stack_feature1, name='stack2',
                                    drop_probability=drop_probability, is_training=is_training)

    # stack_feature2_concat = tf.concat([feature2_pool, stack_feature2], 3)
    stack_feature3 = ae_stack_layer(input_tensor=stack_feature2, name='stack3',
                                    drop_probability=drop_probability, is_training=is_training)

    """ output 56x56"""
    output_stacked1 = tf.layers.conv2d(inputs=stack_feature1, filters=flags.num_of_class, activation=None,
                                       kernel_size=[1, 1], strides=[1, 1], padding='same', name='output_stacked1')

    # output_stacked1_concat = tf.concat([stack_feature2, output_stacked1], 3)
    output_stacked2 = tf.layers.conv2d(inputs=stack_feature2, filters=flags.num_of_class, activation=None,
                                       kernel_size=[1, 1], strides=[1, 1], padding='same', name='output_stacked2')

    # output_stacked2_concat = tf.concat([stack_feature3, output_stacked2], 3)
    output_stacked3 = tf.layers.conv2d(inputs=stack_feature3, filters=flags.num_of_class, activation=None,
                                       kernel_size=[1, 1], strides=[1, 1], padding='same', name='output_stacked3')

    """ bilinear upscale-up 512x512"""
    # output_up = tf.image.resize_nearest_neighbor(output_stacked2, get_shape(x)[1:3])
    output_up = tf.image.resize_bilinear(output_stacked3, get_shape(x)[1:3])

    return [output_stacked1, output_stacked2, output_stacked3],  output_up


def dense_ae_stacked_two(x, flags, drop_probability=0.0, is_training=False):
    """ Original size 448x448"""
    """ bilinear scale-down 224x224"""
    x_down = tf.image.resize_bilinear(x, (get_shape(x)[1] // 2, get_shape(x)[2] // 2))
    x_down = (x_down / 127.5) - 1.0
    """ td1 224x224"""
    feature1 = dense_block_down(input_tensor=x_down, num_layers=4, growth_rate=8, filters=32,
                                name='feature_td1', is_training=is_training, drop_probability=drop_probability)
    # ------------------------------------------------------------------------------------------
    # Use conv instead of resize to learn the useful features
    feature1_pool = pool2d(input_tensor=feature1)
    feature2 = dense_block_down(input_tensor=feature1_pool, num_layers=4, growth_rate=16, filters=64,
                                name='feature_td2', is_training=is_training, drop_probability=drop_probability)
    """ td3 56x56"""
    feature2_pool = pool2d(feature2)
    stack_feature1 = ae_stack_layer(input_tensor=feature2_pool, name='stack1',
                                    drop_probability=drop_probability, is_training=is_training)

    # stack_feature1_concat = tf.concat([feature2_pool, stack_feature1], 3)
    stack_feature2 = ae_stack_layer(input_tensor=stack_feature1, name='stack2',
                                    drop_probability=drop_probability, is_training=is_training)

    # stack_feature2_concat = tf.concat([feature2_pool, stack_feature2], 3)
    # stack_feature3 = ae_stack_layer(input_tensor=stack_feature2, name='stack3',
    #                                 drop_probability=drop_probability, is_training=is_training)

    """ output 56x56"""
    output_stacked1 = tf.layers.conv2d(inputs=stack_feature1, filters=flags.num_of_class, activation=None,
                                       kernel_size=[1, 1], strides=[1, 1], padding='same', name='output_stacked1')

    # output_stacked1_concat = tf.concat([stack_feature2, output_stacked1], 3)
    output_stacked2 = tf.layers.conv2d(inputs=stack_feature2, filters=flags.num_of_class, activation=None,
                                       kernel_size=[1, 1], strides=[1, 1], padding='same', name='output_stacked2')

    # output_stacked2_concat = tf.concat([stack_feature3, output_stacked2], 3)
    # output_stacked3 = tf.layers.conv2d(inputs=stack_feature3, filters=flags.num_of_class, activation=None,
    #                                    kernel_size=[1, 1], strides=[1, 1], padding='same', name='output_stacked3')

    """ bilinear upscale-up 512x512"""
    # output_up = tf.image.resize_nearest_neighbor(output_stacked2, get_shape(x)[1:3])
    output_up = tf.image.resize_bilinear(output_stacked2, get_shape(x)[1:3])

    return [output_stacked1, output_stacked2],  output_up

'''
def fcn(x, flags, drop_probability=0.0, is_training=False):
    def vgg_net(fcn_weights, image):
        layers = (
            'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1',

            'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',

            'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3',
            'relu3_3', 'conv3_4', 'relu3_4', 'pool3',

            'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3',
            'relu4_3', 'conv4_4', 'relu4_4', 'pool4',

            'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3',
            'relu5_3', 'conv5_4', 'relu5_4'
        )

        net = {}
        current = image
        for i, name in enumerate(layers):
            kind = name[:4]
            if kind == 'conv':
                kernels, bias = fcn_weights[i][0][0][0][0]
                # matconvnet: weights are [width, height, in_channels, out_channels]
                # tensorflow: weights are [height, width, in_channels, out_channels]
                kernel_init = tf.constant_initializer(np.transpose(kernels, (1, 0, 2, 3)), dtype=tf.float32)
                bias_init = tf.constant_initializer(bias.reshape(-1), dtype=tf.float32)
                current = tf.layers.conv2d(inputs=current, 
                                           kernel_initializer=kernel_init,
                                           bias_initializer=bias_init,
                                           padding='same', activation=None, name=name)
            elif kind == 'relu':
                current = tf.nn.relu(current, name=name)
            elif kind == 'pool':
                current = pool2d(current)
            net[name] = current

        return net

    """ bilinear down-up 512x512"""
    x_down = tf.image.resize_bilinear(x, (get_shape(x)[1] // 8, get_shape(x)[2] // 8))

    """ 256x256"""
    model_data = scipy.io.loadmat('imagenet-vgg-verydeep-19.mat')
    mean = model_data['normalization'][0][0][0]
    mean_pixel = np.mean(mean, axis=(0, 1))
    weights = np.squeeze(model_data['layers'])
    processed_image = x - mean_pixel
    
    """ 8x8"""
    with tf.variable_scope("inference"):
        image_net = vgg_net(weights, processed_image)
        conv_final_layer = image_net["conv5_3"]

        pool5 = pool2d(conv_final_layer)

        conv6 = conv2d(input_tensor=x, filters=4096, activation=tf.nn.relu, name='conv6')
        relu_dropout6 = tf.nn.dropout(relu6, keep_prob=keep_prob)

        W7 = utils.weight_variable([1, 1, 4096, 4096], name="W7")
        b7 = utils.bias_variable([4096], name="b7")
        conv7 = utils.conv2d_basic(relu_dropout6, W7, b7)
        relu7 = tf.nn.relu(conv7, name="relu7")
        if FLAGS.debug:
            utils.add_activation_summary(relu7)
        relu_dropout7 = tf.nn.dropout(relu7, keep_prob=keep_prob)

        W8 = utils.weight_variable([1, 1, 4096, NUM_OF_CLASSESS], name="W8")
        b8 = utils.bias_variable([NUM_OF_CLASSESS], name="b8")
        conv8 = utils.conv2d_basic(relu_dropout7, W8, b8)
        # annotation_pred1 = tf.argmax(conv8, dimension=3, name="prediction1")

        # now to upscale to actual image size
        deconv_shape1 = image_net["pool4"].get_shape()
        W_t1 = utils.weight_variable([4, 4, deconv_shape1[3].value, NUM_OF_CLASSESS], name="W_t1")
        b_t1 = utils.bias_variable([deconv_shape1[3].value], name="b_t1")
        conv_t1 = utils.conv2d_transpose_strided(conv8, W_t1, b_t1, output_shape=tf.shape(image_net["pool4"]))
        fuse_1 = tf.add(conv_t1, image_net["pool4"], name="fuse_1")

        deconv_shape2 = image_net["pool3"].get_shape()
        W_t2 = utils.weight_variable([4, 4, deconv_shape2[3].value, deconv_shape1[3].value], name="W_t2")
        b_t2 = utils.bias_variable([deconv_shape2[3].value], name="b_t2")
        conv_t2 = utils.conv2d_transpose_strided(fuse_1, W_t2, b_t2, output_shape=tf.shape(image_net["pool3"]))
        fuse_2 = tf.add(conv_t2, image_net["pool3"], name="fuse_2")

        shape = tf.shape(image)
        deconv_shape3 = tf.stack([shape[0], shape[1], shape[2], NUM_OF_CLASSESS])
        W_t3 = utils.weight_variable([16, 16, NUM_OF_CLASSESS, deconv_shape2[3].value], name="W_t3")
        b_t3 = utils.bias_variable([NUM_OF_CLASSESS], name="b_t3")
        conv_t3 = utils.conv2d_transpose_strided(fuse_2, W_t3, b_t3, output_shape=deconv_shape3, stride=8)

        annotation_pred = tf.argmax(conv_t3, dimension=3, name="prediction")

    return tf.expand_dims(annotation_pred, dim=3), conv_t3
'''


def stacked_ae(x, flags, drop_probability=0.0, is_training=False):
    """ bilinear down-up 512x512"""
    x_down = tf.image.resize_bilinear(x, (get_shape(x)[1] // 2, get_shape(x)[2] // 2))
    # ------------------------------------------------------------------------------------------
    # Use conv instead of resize to learn the useful features
    """ td1 256x256"""
    feature1 = dense_block_down(input_tensor=x_down, num_layers=4, growth_rate=8, filters=32,
                                name='feature_td1', is_training=is_training, drop_probability=drop_probability)
    """ td2 128x128"""
    feature1_pool = pool2d(input_tensor=feature1)
    feature2 = dense_block_down(input_tensor=feature1_pool, num_layers=4, growth_rate=8, filters=64,
                                name='feature_td2', is_training=is_training, drop_probability=drop_probability)
    # ------------------------------------------------------------------------------------------
    """ stacked-1 64x64"""
    feature2_pool = pool2d(input_tensor=feature2)
    stacked1 = stacked_layer(input_tensor=feature2_pool, is_training=is_training,
                             drop_probability=drop_probability, name='stacked-1')
    stacked2 = stacked_layer(input_tensor=stacked1, is_training=is_training,
                             drop_probability=drop_probability, name='stacked-2')
    # feature_up4 = feature_up3
    """ output 64x64"""
    output_stacked1 = tf.layers.conv2d(inputs=stacked1, filters=flags.num_of_class, kernel_size=[1, 1],
                                       strides=[1, 1], padding='same', activation=None, name='output_stacked1')
    output_stacked2 = tf.layers.conv2d(inputs=stacked2, filters=flags.num_of_class, kernel_size=[1, 1],
                                       strides=[1, 1], padding='same', activation=None, name='output_stacked2')
    # output_stacked2 = output_stacked1
    """ bilinear upscale-up 512x512"""
    # output_up = tf.image.resize_nearest_neighbor(output_stacked2, get_shape(x)[1:3])
    output_up = tf.image.resize_bilinear(output_stacked2, get_shape(x)[1:3])

    return [output_stacked1, output_stacked2],  output_up


def stuff512_test(x, flags, drop_probability=0.0, is_training=False):
    x_down = tf.image.resize_bilinear(x, (get_shape(x)[1] // 8, get_shape(x)[2] // 8))

    output_stacked2 = tf.layers.conv2d(inputs=x_down, filters=flags.num_of_class, kernel_size=[1, 1],
                                       strides=[1, 1], padding='same', activation=None, name='output_stacked1')
    output_up = tf.image.resize_bilinear(output_stacked2, get_shape(x)[1:3])
    return [output_stacked2, output_stacked2],  output_up


def loss_stacked(x, flags, drop_probability=0.0, is_training=False):
    # ------------------------------------------------------------------------------------------
    # Use conv instead of resize to learn the useful features
    """ td1 256x256"""
    feature1 = dense_block_down(input_tensor=x, num_layers=4, growth_rate=8, filters=32,
                                name='feature_td1', is_training=is_training, drop_probability=drop_probability)
    """ td2 128x128"""
    feature1_pool = pool2d(input_tensor=feature1)
    feature2 = dense_block_down(input_tensor=feature1_pool, num_layers=4, growth_rate=16, filters=64,
                                name='feature_td2', is_training=is_training, drop_probability=drop_probability)
    # ------------------------------------------------------------------------------------------
    """ stacked-1 64x64"""
    feature2_pool = pool2d(input_tensor=feature2)
    stacked1 = stacked_layer(input_tensor=feature2_pool, is_training=is_training,
                             drop_probability=drop_probability, name='stacked-1')
    stacked2 = stacked_layer(input_tensor=stacked1, is_training=is_training,
                             drop_probability=drop_probability, name='stacked-2')
    # feature_up4 = feature_up3
    """ output 64x64"""
    output_stacked1 = tf.layers.conv2d(inputs=stacked1, filters=flags.num_of_class, kernel_size=[1, 1],
                                       strides=[1, 1], padding='same', activation=None, name='output_stacked1')
    output_stacked2 = tf.layers.conv2d(inputs=stacked2, filters=flags.num_of_class, kernel_size=[1, 1],
                                       strides=[1, 1], padding='same', activation=None, name='output_stacked2')
    """ bilinear upscale-up"""
    # output_up = tf.image.resize_nearest_neighbor(output_stacked2, get_shape(x)[1:3])
    output_up = tf.image.resize_bilinear(output_stacked2, get_shape(x)[1:3])

    return [output_stacked1, output_stacked2],  output_up


def stacked_layer_lite(input_tensor, is_training=False, drop_probability=0.0, name='stacked'):
    with tf.variable_scope(name):
        """ td3 64x64"""
        feature3 = dense_block_down(input_tensor=input_tensor, num_layers=4, growth_rate=16, filters=128,
                                    name='feature_td3', is_training=is_training, drop_probability=drop_probability)
        """ td4 32x32"""
        feature3_pool = pool2d(input_tensor=feature3)
        feature4 = dense_block_down(input_tensor=feature3_pool, num_layers=4, growth_rate=16, filters=256,
                                    name='feature_td4', is_training=is_training, drop_probability=drop_probability)
        """ td5 16x16"""
        feature4_pool = pool2d(input_tensor=feature4)
        feature5 = dense_block_down(input_tensor=feature4_pool, num_layers=8, growth_rate=16, filters=256,
                                    name='feature_td5', is_training=is_training, drop_probability=drop_probability)
        # ------------------------------------------------------------------------------------------
        """ bottle_neck 8x8"""
        feature5_pool = pool2d(input_tensor=feature5)
        feature_bottle = dense_block_down(input_tensor=feature5_pool, num_layers=12, growth_rate=16, filters=512,
                                          is_training=is_training, drop_probability=drop_probability,
                                          name='feature_bottle')
        # ------------------------------------------------------------------------------------------
        """ tu5 16x16"""
        feature_up = upscale(feature_bottle)
        feature_concat = tf.concat([feature_up, feature4_pool], 3)
        feature_up5 = dense_block_down(input_tensor=feature_concat, num_layers=8, growth_rate=16, filters=256,
                                       name='feature_up5', is_training=is_training, drop_probability=drop_probability)
        """ tu4 32x32"""
        feature_up = upscale(feature_up5)
        feature_concat = tf.concat([feature_up, feature3_pool], 3)
        feature_up4 = dense_block_down(input_tensor=feature_concat, num_layers=4, growth_rate=16, filters=256,
                                       name='feature_up4', is_training=is_training, drop_probability=drop_probability)
        """ tu3 64x64"""
        feature_up = upscale(feature_up4)
        feature_concat = tf.concat([feature_up, input_tensor], 3)
        feature_up3 = dense_block_down(input_tensor=feature_concat, num_layers=4, growth_rate=16, filters=128,
                                       name='feature_up3', is_training=is_training, drop_probability=drop_probability)
    return feature_up3


def loss_lite(x, flags, drop_probability=0.0, is_training=False):
    # ------------------------------------------------------------------------------------------
    # Use conv instead of resize to learn the useful features
    """ td1 256x256"""
    feature1 = dense_block_down(input_tensor=x, num_layers=4, growth_rate=4, filters=8,
                                name='feature_td1', is_training=is_training, drop_probability=drop_probability)
    """ td2 128x128"""
    feature1_pool = pool2d(input_tensor=feature1)
    feature2 = dense_block_down(input_tensor=feature1_pool, num_layers=4, growth_rate=4, filters=8,
                                name='feature_td2', is_training=is_training, drop_probability=drop_probability)
    # ------------------------------------------------------------------------------------------
    """ stacked-1 64x64"""
    feature2_pool = pool2d(input_tensor=feature2)
    stacked1 = stacked_layer(input_tensor=feature2_pool, is_training=is_training,
                             drop_probability=drop_probability, name='stacked-1')
    stacked2 = stacked_layer(input_tensor=stacked1, is_training=is_training,
                             drop_probability=drop_probability, name='stacked-2')
    # feature_up4 = feature_up3
    """ output 64x64"""
    output_stacked1 = tf.layers.conv2d(inputs=stacked1, filters=flags.num_of_class, kernel_size=[1, 1],
                                       strides=[1, 1], padding='same', activation=None, name='output_stacked1')
    output_stacked2 = tf.layers.conv2d(inputs=stacked2, filters=flags.num_of_class, kernel_size=[1, 1],
                                       strides=[1, 1], padding='same', activation=None, name='output_stacked2')
    """ bilinear upscale-up"""
    # output_up = tf.image.resize_nearest_neighbor(output_stacked2, get_shape(x)[1:3])
    output_up = tf.image.resize_bilinear(output_stacked2, get_shape(x)[1:3])

    return [output_stacked1, output_stacked2],  output_up


def dense_stacked(x, flags, drop_probability=0.0, is_training=False):
    # ------------------------------------------------------------------------------------------
    # Use conv instead of resize to learn the useful features
    """ td1 256x256"""
    feature1 = dense_block_down(input_tensor=x, num_layers=4, growth_rate=16, filters=64,
                                name='feature_td1', is_training=is_training, drop_probability=drop_probability)
    """ td2 128x128"""
    feature1_pool = pool2d(input_tensor=feature1)
    feature2 = dense_block_down(input_tensor=feature1_pool, num_layers=8, growth_rate=16, filters=128,
                                name='feature_td2', is_training=is_training, drop_probability=drop_probability)
    # ------------------------------------------------------------------------------------------
    feature2_pool = pool2d(input_tensor=feature2)
    feature_up3 = stacked_layer(input_tensor=feature2_pool, is_training=is_training,
                                drop_probability=drop_probability, name='stacked-1')
    feature_up4 = stacked_layer(input_tensor=feature_up3, is_training=is_training,
                                drop_probability=drop_probability, name='stacked-2')
    # feature_up4 = feature_up3
    """ output 64x64"""
    output_semi = tf.layers.conv2d(inputs=feature_up3, filters=flags.num_of_class, kernel_size=[1, 1],
                                   strides=[1, 1], padding='same', activation=None, name='output_semi')
    output_final = tf.layers.conv2d(inputs=feature_up4, filters=flags.num_of_class, kernel_size=[1, 1],
                                    strides=[1, 1], padding='same', activation=None, name='output_final')
    output_up = tf.image.resize_nearest_neighbor(output_final, get_shape(x)[1:3])
    return output_semi, output_final,  output_up


def dense_dilated(x, flags, drop_probability=0.0, is_training=False):
    # ------------------------------------------------------------------------------------------
    # Use conv instead of resize to learn the useful features
    """ td1 256x256"""
    feature1 = dense_block_down(input_tensor=x, num_layers=4, growth_rate=16, filters=64,
                                name='feature_td1', is_training=is_training, drop_probability=drop_probability)
    """ td2 128x128"""
    feature1_pool = pool2d(input_tensor=feature1)
    feature2 = dense_block_down(input_tensor=feature1_pool, num_layers=8, growth_rate=16, filters=128,
                                name='feature_td2', is_training=is_training, drop_probability=drop_probability)
    # ------------------------------------------------------------------------------------------
    """ td3 64x64"""
    feature2_pool = pool2d(input_tensor=feature2)
    feature3 = dense_block_down(input_tensor=feature2_pool, num_layers=12, growth_rate=16, filters=256,
                                name='feature_td3', is_training=is_training, drop_probability=drop_probability)
    """ td4 64x64"""
    feature3_dilated = dilated_layer(input_tensor=feature3, filters=256, is_training=is_training,
                                     drop_probability=drop_probability, name='feature_td3_dilated')
    feature4 = dense_block_down(input_tensor=feature3_dilated, num_layers=12, growth_rate=16, filters=512,
                                name='feature_td4', is_training=is_training, drop_probability=drop_probability)
    """ td5 64x64"""
    feature4_dilated = dilated_layer(input_tensor=feature4, filters=512, is_training=is_training,
                                     drop_probability=drop_probability, name='feature_td4_dilated')
    feature5 = dense_block_down(input_tensor=feature4_dilated, num_layers=12, growth_rate=16, filters=512,
                                name='feature_td5', is_training=is_training, drop_probability=drop_probability)
    # ------------------------------------------------------------------------------------------
    """ bottle_neck 64x64"""
    feature5_dilated = dilated_layer(input_tensor=feature5, filters=512, is_training=is_training,
                                     drop_probability=drop_probability, name='feature_td5_dilated')
    feature_bottle = dense_block_down(input_tensor=feature5_dilated, num_layers=16, growth_rate=16, filters=1024,
                                      is_training=is_training, drop_probability=drop_probability, name='feature_bottle')
    """ output 64x64"""
    output = tf.layers.conv2d(inputs=feature_bottle, filters=flags.num_of_class, kernel_size=[1, 1],
                              strides=[1, 1], padding='same', activation=None, name='output')
    output_up = tf.image.resize_nearest_neighbor(output, get_shape(x)[1:3])
    return output, output_up


def dense_super(x, flags, drop_probability=0.0, is_training=False):
    # ------------------------------------------------------------------------------------------
    # Use conv instead of resize to learn the useful features
    """ td1 256x256"""
    feature1 = dense_block_down(input_tensor=x, num_layers=4, growth_rate=16, filters=64,
                                name='feature_td1', is_training=is_training, drop_probability=drop_probability)
    """ td2 128x128"""
    feature1_pool = pool2d(input_tensor=feature1)
    feature2 = dense_block_down(input_tensor=feature1_pool, num_layers=8, growth_rate=16, filters=128,
                                name='feature_td2', is_training=is_training, drop_probability=drop_probability)
    # ------------------------------------------------------------------------------------------
    """ td3 64x64"""
    feature2_pool = pool2d(input_tensor=feature2)
    feature3 = dense_block_down(input_tensor=feature2_pool, num_layers=12, growth_rate=16, filters=256,
                                name='feature_td3', is_training=is_training, drop_probability=drop_probability)
    """ td4 32x32"""
    feature3_pool = pool2d(input_tensor=feature3)
    feature4 = dense_block_down(input_tensor=feature3_pool, num_layers=12, growth_rate=16, filters=512,
                                name='feature_td4', is_training=is_training, drop_probability=drop_probability)
    """ td5 16x16"""
    feature4_pool = pool2d(input_tensor=feature4)
    feature5 = dense_block_down(input_tensor=feature4_pool, num_layers=12, growth_rate=16, filters=512,
                                name='feature_td5', is_training=is_training, drop_probability=drop_probability)
    # ------------------------------------------------------------------------------------------
    """ bottle_neck 8x8"""
    feature5_pool = pool2d(input_tensor=feature5)
    feature_bottle = dense_block_down(input_tensor=feature5_pool, num_layers=16, growth_rate=16, filters=1024,
                                      is_training=is_training, drop_probability=drop_probability, name='feature_bottle')
    # ------------------------------------------------------------------------------------------
    """ tu5 16x16"""
    feature_up = upscale(feature_bottle)
    feature_concat = tf.concat([feature_up, feature4_pool], 3)
    feature_up5 = dense_block_down(input_tensor=feature_concat, num_layers=12, growth_rate=16, filters=512,
                                   name='feature_up5', is_training=is_training, drop_probability=drop_probability)
    """ tu4 32x32"""
    feature_up = upscale(feature_up5)
    feature_concat = tf.concat([feature_up, feature3_pool], 3)
    feature_up4 = dense_block_down(input_tensor=feature_concat, num_layers=12, growth_rate=16, filters=512,
                                   name='feature_up4', is_training=is_training, drop_probability=drop_probability)
    """ tu3 64x64"""
    feature_up = upscale(feature_up4)
    feature_concat = tf.concat([feature_up, feature2_pool], 3)
    feature_up3 = dense_block_down(input_tensor=feature_concat, num_layers=12, growth_rate=16, filters=256,
                                   name='feature_up3', is_training=is_training, drop_probability=drop_probability)
    """ output 64x64"""
    output = tf.layers.conv2d(inputs=feature_up3, filters=flags.num_of_class, kernel_size=[1, 1],
                              strides=[1, 1], padding='same', activation=None, name='output')
    output_up = tf.image.resize_nearest_neighbor(output, get_shape(x)[1:3])
    return output, output_up


def simple_small(x, flags, drop_probability=0.0, is_training=False):
    """ td1 256x256"""
    feature1 = dense_block_down(input_tensor=x, num_layers=4, growth_rate=16, filters=64,
                                name='feature_td1', is_training=is_training, drop_probability=drop_probability)
    """ td2 128x128"""
    feature1_pool = pool2d(input_tensor=feature1)
    feature2 = dense_block_down(input_tensor=feature1_pool, num_layers=8, growth_rate=16, filters=128,
                                name='feature_td2', is_training=is_training, drop_probability=drop_probability)
    """ td3 64x64"""
    feature2_pool = pool2d(input_tensor=feature2)
    feature3 = dense_block_down(input_tensor=feature2_pool, num_layers=16, growth_rate=16, filters=256,
                                name='feature_td3', is_training=is_training, drop_probability=drop_probability)
    """ td4 32x32"""
    feature3_pool = pool2d(input_tensor=feature3)
    feature4 = dense_block_down(input_tensor=feature3_pool, num_layers=16, growth_rate=16, filters=512,
                                name='feature_td4', is_training=is_training, drop_probability=drop_probability)
    """ td5 16x16"""
    feature4_pool = pool2d(input_tensor=feature4)
    feature5 = dense_block_down(input_tensor=feature4_pool, num_layers=16, growth_rate=16, filters=512,
                                name='feature_td5', is_training=is_training, drop_probability=drop_probability)
    """ td6 8x8"""
    feature5_pool = pool2d(input_tensor=feature5)
    feature6 = dense_block_down(input_tensor=feature5_pool, num_layers=16, growth_rate=16, filters=512,
                                name='feature_td6', is_training=is_training, drop_probability=drop_probability)
    # ------------------------------------------------------------------------------------------
    """ bottle_neck 4x4"""
    feature6_pool = pool2d(input_tensor=feature6)
    feature_bottle = dense_block_down(input_tensor=feature6_pool, num_layers=16, growth_rate=16, filters=1024,
                                      is_training=is_training, drop_probability=drop_probability, name='feature_bottle')
    # ------------------------------------------------------------------------------------------

    """ tu6 8x8"""
    feature_up = upscale(feature_bottle)
    feature_concat = tf.concat([feature_up, feature5_pool], 3)
    feature_up6 = dense_block_down(input_tensor=feature_concat, num_layers=16, growth_rate=16, filters=512,
                                   name='feature_up6', is_training=is_training, drop_probability=drop_probability)
    """ tu5 16x16"""
    feature_up = upscale(feature_up6)
    feature_concat = tf.concat([feature_up, feature4_pool], 3)
    feature_up5 = dense_block_down(input_tensor=feature_concat, num_layers=16, growth_rate=16, filters=512,
                                   name='feature_up5', is_training=is_training, drop_probability=drop_probability)
    """ tu4 32x32"""
    feature_up = upscale(feature_up5)
    feature_concat = tf.concat([feature_up, feature3_pool], 3)
    feature_up4 = dense_block_down(input_tensor=feature_concat, num_layers=16, growth_rate=16, filters=512,
                                   name='feature_up4', is_training=is_training, drop_probability=drop_probability)
    """ tu3 64x64"""
    feature_up = upscale(feature_up4)
    feature_concat = tf.concat([feature_up, feature2_pool], 3)
    feature_up3 = dense_block_down(input_tensor=feature_concat, num_layers=16, growth_rate=16, filters=256,
                                   name='feature_up3', is_training=is_training, drop_probability=drop_probability)
    """ tu2 128x128"""
    feature_up = upscale(feature_up3)
    feature_concat = tf.concat([feature_up, feature1_pool], 3)
    feature_up2 = dense_block_down(input_tensor=feature_concat, num_layers=8, growth_rate=16, filters=128,
                                   name='feature_up2', is_training=is_training, drop_probability=drop_probability)
    """ tu1 256x256"""
    feature_up = upscale(feature_up2)
    feature_concat = tf.concat([feature_up, x], 3)
    feature_up1 = dense_block_down(input_tensor=feature_concat, num_layers=4, growth_rate=16, filters=64,
                                   name='feature_up1', is_training=is_training, drop_probability=drop_probability)
    """ output 256x256"""
    output = tf.layers.conv2d(inputs=feature_up1, filters=flags.num_of_class, kernel_size=[1, 1],
                              strides=[1, 1], padding='same', activation=None, name='output')
    return output


def ae(x, flags, is_training=False, drop_probability=0.0):
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