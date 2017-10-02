from __future__ import print_function
import tensorflow as tf
import dataset_parser
from model import simple_256_pool

FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_integer("image_height", "256", "image target height")
tf.flags.DEFINE_integer("image_width", "256", "image target width")
tf.flags.DEFINE_integer("num_of_feature", "3", "number of feature")
tf.flags.DEFINE_integer("num_of_class", "184", "number of class")

tf.flags.DEFINE_string("logs_dir", "./logs_256_pool", "path to logs directory")
tf.flags.DEFINE_integer("num_epochs", "50", "number of epochs for training")
tf.flags.DEFINE_integer("batch_size", "2", "batch size for training")
tf.flags.DEFINE_float("learning_rate", "1e-4", "Learning rate for Adam Optimizer")
tf.flags.DEFINE_string('mode', "train", "Mode train/ test-dev/ test")


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
        # coco_parser.data2record(name='coco_stuff2017_val_10.tfrecords', is_training=False, test_num=10)
        # coco_parser.data2record_test(name='coco_stuff2017_test-dev_all_label.tfrecords', is_dev=True, test_num=None)
        # coco_parser.data2record_test(name='coco_stuff2017_test_all_label.tfrecords', is_dev=False, test_num=None)
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
                name='coco_stuff2017_val_all_label.tfrecords', batch_size=FLAGS.batch_size)
            # A feed-able iterator
            handle = tf.placeholder(tf.string, shape=[])
            iterator = tf.contrib.data.Iterator.from_string_handle(
                handle, training_dataset.output_types, training_dataset.output_shapes)
            next_x, next_y = iterator.get_next()
            # Place_holder
            learning_rate = tf.placeholder(tf.float32)
            is_training = tf.placeholder(tf.bool)
            drop_probability = tf.placeholder(tf.float32, name="drop_probability")
            valid_average_loss = tf.placeholder(tf.float32, name="valid_average_loss")
            data_x = tf.placeholder(tf.float32, shape=[None, None, None, FLAGS.num_of_feature], name="data_x")
        """
        Network (Computes predictions from the inference model)
        """
        with tf.name_scope(name='Network'):
            # Inference
            network_name = 'simple_256_pool'
            with tf.variable_scope(network_name, reuse=False):
                logits = simple_256_pool(
                    x=next_x, flags=FLAGS, is_training=is_training, drop_probability=drop_probability)
            # Use for arbitrary shape
            with tf.variable_scope(network_name, reuse=True):
                logits_infer = simple_256_pool(
                    x=data_x, flags=FLAGS, is_training=False, drop_probability=0.0)
            # Loss
            loss = tf.reduce_mean((tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=logits, labels=next_y, name="loss")))
            # Optimizer
            global_step = tf.Variable(0, trainable=False)
            trainable_var = tf.get_collection(key=tf.GraphKeys.TRAINABLE_VARIABLES, scope=network_name)
            if True:
                import tensorflow.contrib.slim as slim
                slim.model_analyzer.analyze_vars(trainable_var, print_info=True)
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                train_op = tf.train.AdamOptimizer(learning_rate).minimize(
                    loss=loss, global_step=global_step, var_list=trainable_var)
        """
        Other Variables 
        """
        with tf.name_scope(name='Others'):
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
            with tf.name_scope(name='Initial'):
                summary_writer = tf.summary.FileWriter(coco_parser.logs_dir, sess.graph)
                sess.run(init_op)
                ckpt = tf.train.get_checkpoint_state(coco_parser.checkpoint_dir)
                if ckpt and ckpt.model_checkpoint_path:
                    saver.restore(sess, ckpt.model_checkpoint_path)
                    print("Model restored: {}".format(ckpt.model_checkpoint_path))
                else:
                    print("No Model found.")
            """
            Training Mode
            """
            if FLAGS.mode == 'train':
                print('Training mode! Batch size:{:d}, Learning rate:{:f}'.format(
                    FLAGS.batch_size, FLAGS.learning_rate))
                events_freq, save_freq = 100, 1000
                observe_freq, val_freq, val_max_num = 1500, 2000, 200
                """
                Iterators
                """
                with tf.name_scope(name='Iterators'):
                    training_iterator = training_dataset.make_initializable_iterator()
                    validation_iterator = validation_dataset.make_initializable_iterator()
                    training_handle = sess.run(training_iterator.string_handle())
                    validation_handle = sess.run(validation_iterator.string_handle())
                    feed_dict_train = {drop_probability: 0.2, is_training: True, learning_rate: FLAGS.learning_rate,
                                       handle: training_handle}
                    feed_dict_infer = {drop_probability: 0.0, is_training: False,
                                       handle: validation_handle}
                print('Start Training!')
                """
                For each epochs
                """
                for epoch in range(FLAGS.num_epochs):
                    """
                    Start Training
                    """
                    sess.run(training_iterator.initializer)
                    while True:
                        try:
                            # Optimize training loss
                            _, loss_sess, global_step_sess, summary_str, next_x_sess, next_y_sess, logits_sess = \
                                sess.run([train_op, loss, global_step, summary_loss, next_x, next_y, logits],
                                         feed_dict=feed_dict_train)
                            print('[{:d}/{:d}, global_step:{:d}, training loss:{:f}]'.format(
                                epoch, FLAGS.num_epochs, global_step_sess, loss_sess))

                            # Logging the events
                            if global_step_sess % events_freq == 1:
                                summary_writer.add_summary(summary_str, global_step_sess)

                            # Observe training situation (For debugging.)
                            if True and global_step_sess % observe_freq == 1:
                                print('Logging training images.')
                                coco_parser.visualize_data(
                                    x_batches=next_x_sess, y_batches=next_y_sess,
                                    global_step=global_step_sess, pred_batches=logits_sess,
                                    logs_dir=coco_parser.logs_image_train_dir)

                            # Logging the events (validation)
                            if True and global_step_sess % val_freq == (val_freq-1):
                                valid_sum_loss, valid_times = 0, 0
                                sess.run(validation_iterator.initializer)
                                while True:
                                    try:
                                        # Logging the events
                                        loss_sess, next_x_sess, next_y_sess, logits_sess = \
                                            sess.run([loss, next_x, next_y, logits], feed_dict=feed_dict_infer)
                                        print(
                                            'val_step:{:d}, validation loss:{:f}'.format(valid_times, loss_sess))

                                        # Observe validation situation (For debugging.)
                                        if True and valid_times == 0:
                                            # next_x_sess, next_y_sess, logits_sess = \
                                            #     sess.run([next_x, next_y, logits], feed_dict=feed_dict_infer)
                                            print('Logging validation images.')
                                            coco_parser.visualize_data(
                                                x_batches=next_x_sess, y_batches=next_y_sess,
                                                global_step=global_step_sess, pred_batches=logits_sess,
                                                logs_dir=coco_parser.logs_image_valid_dir)

                                        valid_sum_loss += loss_sess
                                        valid_times += 1
                                        if valid_times > val_max_num:
                                            break
                                    except tf.errors.OutOfRangeError:
                                        break

                                # Calculate and store score.
                                valid_average = valid_sum_loss / valid_times
                                print('-----Validation finished! Average validation loss:{:f}-----'.format(
                                    valid_average))
                                summary_str = sess.run(summary_loss_valid,
                                                       feed_dict={valid_average_loss: valid_average})
                                summary_writer.add_summary(summary_str, global_step_sess)
                            """
                            Saving the checkpoint
                            """
                            if global_step_sess % save_freq == 0:
                                print('Saving model...')
                                saver.save(sess, coco_parser.checkpoint_dir + '/model.ckpt',
                                           global_step=global_step_sess)

                        except tf.errors.OutOfRangeError:
                            print('----------------One epochs finished!----------------')
                            break

            elif FLAGS.mode == 'test-dev':
                coco_parser.inference_with_tf(sess=sess, logits_infer=logits_infer, data_x=data_x,
                                              drop_probability=drop_probability, is_training=is_training,
                                              is_dev=True, pool_times=4)

            elif FLAGS.mode == 'test':
                coco_parser.inference_with_tf(sess=sess, logits_infer=logits_infer, data_x=data_x,
                                              drop_probability=drop_probability, is_training=is_training,
                                              is_dev=False, pool_times=4)

if __name__ == "__main__":
    tf.app.run()
