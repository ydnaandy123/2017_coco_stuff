from __future__ import print_function
import scipy.misc
import tensorflow as tf
import scipy.io as sio
import numpy as np
import dataset_parser

FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_integer("image_height", "256", "image target height")
tf.flags.DEFINE_integer("image_width", "256", "image target width")
tf.flags.DEFINE_integer("num_of_feature", "3", "number of feature")
tf.flags.DEFINE_integer("num_of_class", "1", "number of class")

tf.flags.DEFINE_string("logs_dir", "./logs_ae_elec", "path to logs directory")
tf.flags.DEFINE_integer("epochs", "30", "epochs for training")
tf.flags.DEFINE_integer("batch_size", "9", "batch size for training")
tf.flags.DEFINE_float("learning_rate", "1e-4", "Learning rate for Adam Optimizer")
tf.flags.DEFINE_string('mode', "train", "Mode train/ test/ visualize")


def get_shape(tensor):
    static_shape = tensor.shape.as_list()
    dynamic_shape = tf.unstack(tf.shape(tensor))
    dims = [s[1] if s[0] is None else s[0]
            for s in zip(static_shape, dynamic_shape)]
    return dims


def simple_ae(x, drop_probability, is_training=False):
    with tf.variable_scope("simple_ae"):
        """ conv0 256x256"""
        conv0 = tf.layers.conv2d(inputs=x, filters=32, kernel_size=[3, 3],
                                 strides=[1, 1], padding='same', activation=tf.nn.relu, name='conv0')
        """ conv1 256x256"""
        conv1 = tf.layers.conv2d(inputs=conv0, filters=64, kernel_size=[3, 3],
                                 strides=[1, 1], padding='same', activation=tf.nn.relu, name='conv1')
        batch_c1 = tf.layers.batch_normalization(inputs=conv1, training=is_training, name='batch_c1')
        pool1 = tf.layers.max_pooling2d(inputs=batch_c1, pool_size=[2, 2], strides=[2, 2])
        """ conv2 128x128"""
        conv2 = tf.layers.conv2d(inputs=pool1, filters=128, kernel_size=[3, 3],
                                 strides=[1, 1], padding='same', activation=tf.nn.relu, name='conv2')
        batch_c2 = tf.layers.batch_normalization(inputs=conv2, training=is_training, name='batch_c2')
        pool2 = tf.layers.max_pooling2d(inputs=batch_c2, pool_size=[2, 2], strides=[2, 2])
        """ conv3 64x64"""
        conv3 = tf.layers.conv2d(inputs=pool2, filters=256, kernel_size=[3, 3],
                                 strides=[1, 1], padding='same', activation=tf.nn.relu, name='conv3')
        batch_c3 = tf.layers.batch_normalization(inputs=conv3, training=is_training, name='batch_c3')
        pool3 = tf.layers.max_pooling2d(inputs=batch_c3, pool_size=[2, 2], strides=[2, 2])
        """ conv4 32x32"""
        conv4 = tf.layers.conv2d(inputs=pool3, filters=512, kernel_size=[3, 3],
                                 strides=[1, 1], padding='same', activation=tf.nn.relu, name='conv4')
        batch_c4 = tf.layers.batch_normalization(inputs=conv4, training=is_training, name='batch_c4')
        pool4 = tf.layers.max_pooling2d(inputs=batch_c4, pool_size=[2, 2], strides=[2, 2])
        """ conv5 16x16"""
        conv5 = tf.layers.conv2d(inputs=pool4, filters=512, kernel_size=[3, 3],
                                 strides=[1, 1], padding='same', activation=tf.nn.relu, name='conv5')
        batch_c5 = tf.layers.batch_normalization(inputs=conv5, training=is_training, name='batch_c5')
        pool5 = tf.layers.max_pooling2d(inputs=batch_c5, pool_size=[2, 2], strides=[2, 2])
        """ conv6 8x8"""
        conv6 = tf.layers.conv2d(inputs=pool5, filters=512, kernel_size=[3, 3],
                                 strides=[1, 1], padding='same', activation=tf.nn.relu, name='conv6')
        batch_c6 = tf.layers.batch_normalization(inputs=conv6, training=is_training, name='batch_c6')
        pool6 = tf.layers.max_pooling2d(inputs=batch_c6, pool_size=[2, 2], strides=[2, 2])
        """ conv7 4x4"""
        conv7 = tf.layers.conv2d(inputs=pool6, filters=512, kernel_size=[3, 3],
                                 strides=[1, 1], padding='same', activation=tf.nn.relu, name='conv7')
        batch_c7 = tf.layers.batch_normalization(inputs=conv7, training=is_training, name='batch_c7')
        pool7 = tf.layers.max_pooling2d(inputs=batch_c7, pool_size=[2, 2], strides=[2, 2])
        """ conv8 2x2"""
        conv8 = tf.layers.conv2d(inputs=pool7, filters=512, kernel_size=[3, 3],
                                 strides=[1, 1], padding='same', activation=tf.nn.relu, name='conv8')
        batch_c8 = tf.layers.batch_normalization(inputs=conv8, training=is_training, name='batch_c8')

        """
        ---------------------------------------------------------------------------------------
        """

        """ deconv7 4x4"""
        deconv7 = tf.layers.conv2d_transpose(inputs=batch_c8, filters=512, kernel_size=[3, 3],
                                             strides=[2, 2], padding='same', activation=tf.nn.relu, name='deconv7')
        batch_d7 = tf.layers.batch_normalization(inputs=deconv7, training=is_training, name='batch_d7')
        deconv7_drop = tf.layers.dropout(inputs=batch_d7, rate=drop_probability,
                                         training=is_training, name='deconv7_drop')
        concat7 = tf.concat([deconv7_drop, pool6], 3)
        """ deconv6 8x8"""
        deconv6 = tf.layers.conv2d_transpose(inputs=concat7, filters=512, kernel_size=[3, 3],
                                             strides=[2, 2], padding='same', activation=tf.nn.relu, name='deconv6')
        batch_d6 = tf.layers.batch_normalization(inputs=deconv6, training=is_training, name='batch_d6')
        deconv6_drop = tf.layers.dropout(inputs=batch_d6, rate=drop_probability,
                                         training=is_training, name='deconv6_drop')
        concat6 = tf.concat([deconv6_drop, pool5], 3)
        """ deconv5 16x16"""
        deconv5 = tf.layers.conv2d_transpose(inputs=concat6, filters=512, kernel_size=[3, 3],
                                             strides=[2, 2], padding='same', activation=tf.nn.relu, name='deconv5')
        batch_d5 = tf.layers.batch_normalization(inputs=deconv5, training=is_training, name='batch_d5')
        deconv5_drop = tf.layers.dropout(inputs=batch_d5, rate=drop_probability,
                                         training=is_training, name='deconv5_drop')
        concat5 = tf.concat([deconv5_drop, pool4], 3)
        """ deconv4 32x32"""
        deconv4 = tf.layers.conv2d_transpose(inputs=concat5, filters=256, kernel_size=[3, 3],
                                             strides=[2, 2], padding='same', activation=tf.nn.relu, name='deconv4')
        batch_d4 = tf.layers.batch_normalization(inputs=deconv4, training=is_training, name='batch_d4')
        concat4 = tf.concat([batch_d4, pool3], 3)
        """ deconv3 64x64"""
        deconv3 = tf.layers.conv2d_transpose(inputs=concat4, filters=128, kernel_size=[3, 3],
                                             strides=[2, 2], padding='same', activation=tf.nn.relu, name='deconv3')
        batch_d3 = tf.layers.batch_normalization(inputs=deconv3, training=is_training, name='batch_d3')
        concat3 = tf.concat([batch_d3, pool2], 3)
        """ deconv2 128x128"""
        deconv2 = tf.layers.conv2d_transpose(inputs=concat3, filters=64, kernel_size=[3, 3],
                                             strides=[2, 2], padding='same', activation=tf.nn.relu, name='deconv2')
        batch_d2 = tf.layers.batch_normalization(inputs=deconv2, training=is_training, name='batch_d2')
        concat2 = tf.concat([batch_d2, pool1], 3)
        """ deconv1 256x256"""
        deconv1 = tf.layers.conv2d_transpose(inputs=concat2, filters=32, kernel_size=[3, 3],
                                             strides=[2, 2], padding='same', activation=tf.nn.relu, name='deconv1')
        batch_d1 = tf.layers.batch_normalization(inputs=deconv1, training=is_training, name='batch_d1')
        """ output 256x256"""
        output = tf.layers.conv2d(inputs=batch_d1, filters=FLAGS.num_of_class, kernel_size=[3, 3],
                                  strides=[1, 1], padding='same', activation=None, name='output')
    return output


def main(args=None):
    print(args)
    tf.reset_default_graph()
    """
    Dataset Parser
    """    # Parse Dataset
    coco_parser = dataset_parser.ELECParser('./dataset/elec',
                                              target_height=FLAGS.image_height, target_width=FLAGS.image_width)
    coco_parser.load_train_paths()
    # Hyper-parameters
    epochs, batch_size = FLAGS.epochs, FLAGS.batch_size
    data_len = len(coco_parser.train_paths)
    print(data_len)
    batches = data_len // batch_size
    """
    Build Graph
    """
    global_step = tf.Variable(0, trainable=False)
    # Placeholder
    learning_rate = tf.placeholder(tf.float32)
    is_training = tf.placeholder(tf.bool)
    drop_probability = tf.placeholder(tf.float32, name="drop_probability")
    data_x = tf.placeholder(tf.float32, shape=[None, None, None, FLAGS.num_of_feature],
                            name="data_x")
    data_y = tf.placeholder(tf.float32, shape=[None, None, None],
                            name="data_y")
    """
    Network
    """
    logits = simple_ae(x=data_x, drop_probability=drop_probability, is_training=is_training)
    # Loss
    loss = tf.reduce_mean(
        tf.losses.mean_squared_error(labels=data_y, predictions=tf.squeeze(logits, axis=3)), name='MSE')
    # loss = tf.reduce_mean((tf.nn.sparse_softmax_cross_entropy_with_logits(
    #     logits=logits, labels=data_y, name="entropy")))
    """
    Optimizer
    """
    trainable_var = tf.get_collection(key=tf.GraphKeys.TRAINABLE_VARIABLES, scope='simple_ae')
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_op = tf.train.AdamOptimizer(learning_rate).minimize(
            loss=loss, global_step=global_step, var_list=trainable_var)
    """
    Graph Logs
    """
    tf.summary.scalar("MSE", loss)
    summary_op = tf.summary.merge_all()
    saver = tf.train.Saver(max_to_keep=2)
    """
    Launch Session
    """
    with tf.Session() as sess:
        summary_writer = tf.summary.FileWriter(FLAGS.logs_dir + '/events', sess.graph)
        sess.run(tf.global_variables_initializer())
        ckpt = tf.train.get_checkpoint_state(FLAGS.logs_dir + '/model')
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            print("Model restored: {}".format(ckpt.model_checkpoint_path))
        else:
            print("No Model found.")

        if FLAGS.mode == 'train':
            cur_learning_rate = FLAGS.learning_rate
            for epoch in range(0, epochs):
                np.random.shuffle(coco_parser.train_paths)
                for batch in range(0, batches):
                    x_batch, y_batch = coco_parser.load_train_datum_batch_aug(batch*batch_size, (batch+1)*batch_size)
                    x_batch_ori = np.array(x_batch, dtype=np.float32)
                    x_batch = x_batch_ori - 127.5
                    y_batch_ori = np.array(y_batch, dtype=np.float32)
                    y_batch = y_batch_ori / 127.5 - 1.0
                    feed_dict = {data_x: x_batch, data_y: y_batch,
                                 drop_probability: 0.2, is_training: True, learning_rate: cur_learning_rate}
                    _, loss_sess, global_step_sess = sess.run([train_op, loss, global_step], feed_dict=feed_dict)

                    print('global_setp: {:d}, epoch: [{:d}/{:d}], batch: [{:d}/{:d}], data: {:d}-{:d}, loss: {:f}'
                          .format(global_step_sess, epoch, epochs, batch, batches,
                                  batch*batch_size, (batch+1)*batch_size, loss_sess))

                    if global_step_sess % 10 == 1:
                        summary_str = sess.run(summary_op, feed_dict={
                            data_x: x_batch, data_y: y_batch, drop_probability: 0.0, is_training: False})
                        summary_writer.add_summary(summary_str, global_step_sess)

                    if global_step_sess % 1000 == 1:
                        logits_sess = sess.run(logits, feed_dict={
                            data_x: x_batch, drop_probability: 0.0, is_training: False})
                        print('Logging images..')
                        for batch_idx, train_path in \
                                enumerate(coco_parser.train_paths[batch*batch_size:(batch+1)*batch_size]):
                            name = train_path[0].split('/')[-1].split('.')[0]
                            scipy.misc.imsave('{}/images/{:d}_{}_0_rgb.png'.format(
                                FLAGS.logs_dir, global_step_sess, name), x_batch_ori[batch_idx].astype(np.uint8))
                            # scipy.misc.imsave('{}/images/{:d}_{}_2_d.png'.format(
                            #     FLAGS.logs_dir, global_step_sess, name), x_batch[batch_idx, :, :, 4])
                            scipy.misc.imsave('{}/images/{:d}_{}_3_gt.png'.format(
                                FLAGS.logs_dir, global_step_sess, name), y_batch_ori[batch_idx].astype(np.uint8))
                            pred_reverse = (np.squeeze(logits_sess[batch_idx], axis=2) + 1.0) * 127.5
                            pred_reverse = np.minimum(np.maximum(pred_reverse, 0), 255)
                            scipy.misc.imsave('{}/images/{:d}_{}_4_pred.png'.format(
                                FLAGS.logs_dir, global_step_sess, name), pred_reverse.astype(np.uint8))

                    if global_step_sess % 2500 == 0:
                        print('Saving model...')
                        saver.save(sess, FLAGS.logs_dir + "/model/model.ckpt", global_step=global_step_sess)


if __name__ == "__main__":
    tf.app.run()
