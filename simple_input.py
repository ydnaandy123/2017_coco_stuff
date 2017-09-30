from __future__ import print_function
import os
import tensorflow as tf
import numpy as np
import dataset_parser
from PIL import Image
from pycocotools.coco import COCO

FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_integer("image_height", "400", "image target height")
tf.flags.DEFINE_integer("image_width", "400", "image target width")
tf.flags.DEFINE_integer("num_of_feature", "3", "number of feature")
tf.flags.DEFINE_integer("num_of_class", "92", "number of class")

tf.flags.DEFINE_string("logs_dir", "./logs_input", "path to logs directory")
tf.flags.DEFINE_integer("num_epochs", "4", "number of epochs for training")
tf.flags.DEFINE_integer("batch_size", "9", "batch size for training")
tf.flags.DEFINE_float("learning_rate", "1e-4", "Learning rate for Adam Optimizer")
tf.flags.DEFINE_string('mode', "train", "Mode train/ test/ visualize")


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


def simple_deconv(x, is_training=False):
    with tf.variable_scope("simple_deconv"):
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
        output = tf.layers.conv2d(inputs=concat1, filters=FLAGS.num_of_class, kernel_size=[3, 3],
                                  strides=[1, 1], padding='same', activation=None, name='output')
    return output


def main(args=None):
    print(args)
    tf.reset_default_graph()
    """
    Read coco parser     
    """
    coco_parser = dataset_parser.MSCOCOParser(dataset_dir='./dataset/coco_stuff',
                                              image_height=FLAGS.image_height, image_width=FLAGS.image_width)
    coco_parser.load_train_paths()
    coco_parser.load_val_paths()
    """
    Transform mscoco to TFRecord format (Only do once.)     
    """
    if False:
        coco_parser.data2record_train(name='coco_stuff2017_train.tfrecords')
        coco_parser.data2record_val(name='coco_stuff2017_val.tfrecords')
        return
    """
    Parse TFRecord format (For test.)
    """
    if True:
        # training_filenames = \
        #     ['/home/andy/Github/2017_coco_stuff/dataset/coco_stuff_TFRecord/coco_stuff2017_train.tfrecords']
        # validation_filenames =\
        #     ['/home/andy/Github/2017_coco_stuff/dataset/coco_stuff_TFRecord/coco_stuff2017_val.tfrecords']
        # filenames = tf.placeholder(tf.string, shape=[None])

        training_iterator = coco_parser.tfrecord_get_iterator(
            name='coco_stuff2017_train.tfrecords', batch_size=FLAGS.batch_size)
        next_training_x, next_training_y = training_iterator.get_next()

        validation_iterator = coco_parser.tfrecord_get_iterator(
            name='coco_stuff2017_val.tfrecords', batch_size=FLAGS.batch_size)
        next_validation_x, next_validation_y = validation_iterator.get_next()

        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        with tf.Session() as sess:
            sess.run(init_op)
            global_step_sess = 0
            for epoch in range(FLAGS.num_epochs):
                sess.run(training_iterator.initializer)
                while True:
                    try:
                        print('[{:d}/{:d}, global_step:{:d}]'.format(epoch, FLAGS.num_epochs, global_step_sess))
                        next_training_x_sess, next_training_y_sess = \
                            sess.run([next_training_x, next_training_y])

                        for batch_idx, next_training_x_sess_single in enumerate(next_training_x_sess):
                            x_png = Image.fromarray(next_training_x_sess_single.astype(np.uint8)).convert('P')
                            x_png.save('{}/images/{:d}_{:d}_0_rgb.png'.format(
                                FLAGS.logs_dir, global_step_sess, batch_idx), format='PNG')

                        for batch_idx, next_training_y_sess_single in enumerate(next_training_y_sess):
                            y_reverse = np.array(next_training_y_sess_single) + 92
                            y_png = Image.fromarray(y_reverse.astype(np.uint8)).convert('P')
                            y_png.putpalette(list(coco_parser.cmap))
                            y_png.save('{}/images/{:d}_{:d}_1_gt.png'.format(
                                FLAGS.logs_dir, global_step_sess, batch_idx), format='PNG')
                        global_step_sess += 1
                        break
                    except tf.errors.OutOfRangeError:
                        print('One epochs finished!')
                        break
                """
                Validation
                """
                sess.run(validation_iterator.initializer)
                next_validation_x_sess, next_validation_y_sess = \
                    sess.run([next_validation_x, next_validation_y])
                x_png = Image.fromarray(next_validation_x_sess[0].astype(np.uint8)).convert('P')
                #x_png.save('{}/images/{}_{:d}_0_rgb.png'.format(
                #    FLAGS.logs_dir, 'val', global_step), format='PNG')
                # break
        return

    with tf.Graph().as_default():
        # filenames = tf.placeholder(tf.string, shape=[None])
        file_names = 'coco_stuff2017.tfrecords'
        dataset = tf.contrib.data.TFRecordDataset(file_names)
        dataset = dataset.map(parser)
        dataset = dataset.shuffle(buffer_size=500)
        dataset = dataset.batch(FLAGS.batch_size)
        #dataset = dataset.repeat(FLAGS.num_epochs)
        # iterator = dataset.make_one_shot_iterator()
        iterator = dataset.make_initializable_iterator()
        # Input images and labels.
        next_example, next_label = iterator.get_next()
        """
        Build Graph
        """
        global_step = tf.Variable(0, trainable=False)
        learning_rate = tf.placeholder(tf.float32)
        is_training = tf.placeholder(tf.bool)
        drop_probability = tf.placeholder(tf.float32, name="drop_probability")
        logits = simple_deconv(x=next_example, is_training=is_training)
        # Loss
        loss = tf.reduce_mean((tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=logits, labels=next_label, name="loss")))
        loss_valid = tf.reduce_mean((tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=logits, labels=next_label, name="loss_valid")))
        """
        Optimizer
        """
        trainable_var = tf.get_collection(key=tf.GraphKeys.TRAINABLE_VARIABLES, scope='simple_deconv')
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_op = tf.train.AdamOptimizer(learning_rate).minimize(
                loss=loss, global_step=global_step, var_list=trainable_var)
        """
        Graph Logs
        """
        summary_loss = tf.summary.scalar("loss", loss)
        summary_loss_valid = tf.summary.scalar("loss_valid", loss_valid)
        saver = tf.train.Saver(max_to_keep=2)
        """
        Initial OP
        """
        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        """
        Session
        """
        with tf.Session() as sess:
            #summary_writer = tf.summary.FileWriter(FLAGS.logs_dir + '/events', sess.graph)
            #sess.run(init_op)
            #ckpt = tf.train.get_checkpoint_state(FLAGS.logs_dir + '/model')
            #if ckpt and ckpt.model_checkpoint_path:
            #    saver.restore(sess, ckpt.model_checkpoint_path)
            #    print("Model restored: {}".format(ckpt.model_checkpoint_path))
            #else:
            #    print("No Model found.")

            sess.run(init_op)

            if FLAGS.mode == 'train':
                sess.run(iterator.initializer)
                while True:
                    try:
                        feed_dict = {drop_probability: 0.2, is_training: True, learning_rate: FLAGS.learning_rate}
                        _, loss_sess, global_step_sess = sess.run([train_op, loss, global_step], feed_dict=feed_dict)

                        print('global_setp: {:d}, loss: {:f}'
                              .format(global_step_sess, loss_sess))

                        if global_step_sess % 20 == 1:
                            logits_sess = next_example.eval()
                            yo = logits_sess

                    except tf.errors.OutOfRangeError:
                        break



            elif FLAGS.mode == 'test-dev':
                print('test')
                # Define path
                ann_file = './dataset/coco_stuff/annotations/image_info_test-dev2017.json'
                test_dir = './dataset/coco_stuff/test2017'
                # Initialize COCO ground truth API
                coco_gt = COCO(ann_file)
                for key_idx, key in enumerate(coco_gt.imgs):
                    print('{:d}/{:d}'.format(key_idx, len(coco_gt.imgs)))
                    value = coco_gt.imgs[key]
                    file_name = value['file_name']
                    image = Image.open(os.path.join(test_dir, file_name))
                    ##############################################################
                    width, height = image.size
                    width_new = ((width // 16) + 1) * 16 if width % 16 != 0 else width
                    height_new = ((height // 16) + 1) * 16 if height % 16 != 0 else height

                    new_im = Image.new("RGB", (width_new, height_new))
                    box_left = np.floor((width_new - width) / 2).astype(np.int32)
                    box_upper = np.floor((height_new - height) / 2).astype(np.int32)
                    new_im.paste(image, (box_left, box_upper))
                    image = new_im
                    # image = image.resize((width_new, height_new), resample=Image.BILINEAR)
                    ##############################################################
                    image = np.array(image)
                    if len(image.shape) < 3:
                        image = np.dstack((image, image, image))
                    image = np.expand_dims(image, axis=0)

                    logits_sess = sess.run(logits, feed_dict={
                        data_x: image, drop_probability: 0.0, is_training: False})

                    pred_reverse = np.argmax(logits_sess[0], axis=2) + 92
                    pred_png = Image.fromarray(pred_reverse.astype(np.uint8)).convert('P')
                    ##############################################################
                    pred_png = pred_png.crop((box_left, box_upper, width, height))
                    # pred_png = pred_png.resize((width, height), resample=Image.NEAREST)
                    ##############################################################
                    pred_png.putpalette(list(coco_parser.cmap))
                    pred_png.save('{}/test-dev/{}'.format(
                        FLAGS.logs_dir, file_name.replace('.jpg', '.png')), format='PNG')

if __name__ == "__main__":
    tf.app.run()
