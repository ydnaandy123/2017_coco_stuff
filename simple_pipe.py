from __future__ import print_function
import os
import tensorflow as tf
import numpy as np
import dataset_parser
from PIL import Image
from pycocotools.coco import COCO
from model import simple_ae

FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_integer("image_height", "400", "image target height")
tf.flags.DEFINE_integer("image_width", "400", "image target width")
tf.flags.DEFINE_integer("num_of_feature", "3", "number of feature")
tf.flags.DEFINE_integer("num_of_class", "184", "number of class")

tf.flags.DEFINE_string("logs_dir", "./logs_pipe", "path to logs directory")
tf.flags.DEFINE_integer("num_epochs", "50", "number of epochs for training")
tf.flags.DEFINE_integer("batch_size", "9", "batch size for training")
tf.flags.DEFINE_float("learning_rate", "1e-4", "Learning rate for Adam Optimizer")
tf.flags.DEFINE_string('mode', "train", "Mode train/ test/ visualize")


def main(args=None):
    print(args)
    tf.reset_default_graph()
    """
    Read coco parser     
    """
    coco_parser = dataset_parser.MSCOCOParser(dataset_dir='./dataset/coco_stuff', flags=FLAGS)
    coco_parser.load_train_paths()
    coco_parser.load_val_paths()
    """
    Transform mscoco to TFRecord format (Only do once.)     
    """
    if False:
        # coco_parser.data2record(name='coco_stuff2017_train_all_label.tfrecords', is_training=True, test_num=None)
        # coco_parser.data2record(name='coco_stuff2017_val_all_label.tfrecords', is_training=False, test_num=None)
        coco_parser.data2record_test(name='coco_stuff2017_test-dev_all_label.tfrecords', is_dev=True, test_num=None)
        coco_parser.data2record_test(name='coco_stuff2017_test_all_label.tfrecords', is_dev=False, test_num=None)
        return
    """
    Build Graph
    """
    with tf.Graph().as_default():
        """
        Input (TFRecord and Place_holder)
        """
        with tf.name_scope(name='Input'):
            # Dataset
            training_dataset = coco_parser.tfrecord_get_dataset(
                name='coco_stuff2017_train_all_label.tfrecords', batch_size=FLAGS.batch_size,
                shuffle_size=None)
            validation_dataset = coco_parser.tfrecord_get_dataset(
                name='coco_stuff2017_val.tfrecords', batch_size=FLAGS.batch_size)

            handle = tf.placeholder(tf.string, shape=[])
            iterator = tf.contrib.data.Iterator.from_string_handle(
                handle, training_dataset.output_types, training_dataset.output_shapes)
            next_element = iterator.get_next()
            iterator = tf.contrib.data.Iterator.from_structure(
                training_dataset.output_types, training_dataset.output_shapes)
            next_x, next_y = iterator.get_next()
            training_init_op = iterator.make_initializer(training_dataset)
            validation_init_op = iterator.make_initializer(validation_dataset)
            # Place_holder
            learning_rate = tf.placeholder(tf.float32)
            is_training = tf.placeholder(tf.bool)
            drop_probability = tf.placeholder(tf.float32, name="drop_probability")
            valid_average_loss = tf.placeholder(tf.float32, name="valid_average_loss")
            data_x = tf.placeholder(tf.float32, shape=[None, None, None, FLAGS.num_of_feature], name="data_x")
            data_y = tf.placeholder(tf.int32, shape=[None, None, None], name="data_y")
        """
        Network (Computes predictions from the inference model)
        """
        with tf.name_scope(name='Network'):
            # Inference
            with tf.variable_scope("simple_deconv", reuse=False):
                logits = simple_ae(x=next_x, flags=FLAGS, is_training=is_training)
            # Loss
            loss = tf.reduce_mean((tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=logits, labels=next_y, name="loss")))
            # Optimizer
            global_step = tf.Variable(0, trainable=False)
            trainable_var = tf.get_collection(key=tf.GraphKeys.TRAINABLE_VARIABLES, scope='simple_deconv')
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                train_op = tf.train.AdamOptimizer(learning_rate).minimize(
                    loss=loss, global_step=global_step, var_list=trainable_var)
        """
        Other Variables 
        """
        with tf.name_scope(name='Other'):
            # Graph Logs
            summary_loss = tf.summary.scalar("loss_train", loss)
            summary_loss_valid = tf.summary.scalar("loss_valid", valid_average_loss)
            saver = tf.train.Saver(max_to_keep=2)
            # Initial OP
            init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        """
        Session
        """
        with tf.Session() as sess:
            """
            Model restore and initialize
            """
            summary_writer = tf.summary.FileWriter(coco_parser.logs_dir, sess.graph)
            sess.run(init_op)
            ckpt = tf.train.get_checkpoint_state(coco_parser.checkpoint_dir)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
                print("Model restored: {}".format(ckpt.model_checkpoint_path))
            else:
                print("No Model found.")
            """
            Select Mode
            """
            if FLAGS.mode == 'train':
                print('Training mode! Batch size is:{:d}'.format(FLAGS.batch_size))
                feed_dict_train = {drop_probability: 0.2, is_training: True, learning_rate: FLAGS.learning_rate}
                feed_dict_infer = {drop_probability: 1.0, is_training: False}
                """
                For each epochs
                """
                for epoch in range(FLAGS.num_epochs):
                    """
                    Training
                    """
                    sess.run(training_init_op)
                    while True:
                        try:
                            """
                            Optimize training loss
                            """
                            _, loss_sess, global_step_sess, summary_str, next_x_sess, next_y_sess, logits_sess = \
                                sess.run([train_op, loss, global_step, summary_loss, next_x, next_y, logits],
                                         feed_dict=feed_dict_train)
                            print('[{:d}/{:d}, global_step:{:d}, training loss:{:f}]'.format(
                                epoch, FLAGS.num_epochs, global_step_sess, loss_sess))
                            """
                            Logging the events
                            """
                            if global_step_sess % 20 == 1:
                                summary_writer.add_summary(summary_str, global_step_sess)
                            """
                            Saving the checkpoint
                            """
                            if global_step_sess % 1000 == 0:
                                print('Saving model...')
                                saver.save(sess, coco_parser.checkpoint_dir + '/model.ckpt',
                                           global_step=global_step_sess)
                            """
                            Observe training situation (For debugging.)
                            """
                            if True and global_step_sess % 1500 == 1:
                                print('Logging training images.')
                                coco_parser.visualize_data(
                                    x_batches=next_x_sess, y_batches=next_y_sess,
                                    global_step=global_step_sess, pred_batches=logits_sess,
                                    logs_dir=coco_parser.logs_image_train_dir)

                        except tf.errors.OutOfRangeError:
                            print('----------------One epochs finished!----------------')
                            break
                    """
                    Validation
                    """
                    sess.run(validation_init_op)
                    valid_sum_loss, valid_times = 0, 0
                    while True:
                        try:
                            """
                            Logging the events
                            """
                            loss_sess, next_x_sess, next_y_sess, logits_sess =\
                                sess.run([loss, next_x, next_y, logits], feed_dict=feed_dict_infer)
                            print('validation_step:{:d}, validation loss:{:f}'.format(valid_times, loss_sess))
                            """
                            Observe validation situation (For debugging.)
                            """
                            if True and valid_times % 500 == 0:
                                # next_x_sess, next_y_sess, logits_sess = \
                                #     sess.run([next_x, next_y, logits], feed_dict=feed_dict_infer)
                                print('Logging training images.')
                                coco_parser.visualize_data(
                                    x_batches=next_x_sess, y_batches=next_y_sess,
                                    global_step=global_step_sess, pred_batches=logits_sess,
                                    logs_dir=coco_parser.logs_image_valid_dir)

                            valid_sum_loss += loss_sess
                            valid_times += 1

                        except tf.errors.OutOfRangeError:
                            valid_average = valid_sum_loss / valid_times
                            print('-----Validation finished! Average validation loss:{:f}-----'.format(valid_average))
                            summary_str = sess.run(summary_loss_valid, feed_dict={valid_average_loss: valid_average})
                            summary_writer.add_summary(summary_str, global_step_sess)
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
