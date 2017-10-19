import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.examples.tutorials.mnist import input_data
from model_new import DenseNet
from dataset_parser import MNISTParser

FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_integer("image_height", "448", "image target height")
tf.flags.DEFINE_integer("image_width", "448", "image target width")
tf.flags.DEFINE_integer("feature_num", "3", "number of feature")
tf.flags.DEFINE_integer("class_num", "10", "number of class")

# Hyper-parameter
tf.flags.DEFINE_integer("growth_k", "12", "growth_k")
tf.flags.DEFINE_integer("nb_block", "2", "how many (dense block + Transition Layer)")
tf.flags.DEFINE_float("init_learning_rate", "1e-4", "Learning rate for Adam Optimizer")
tf.flags.DEFINE_float("epsilon", "1e-8", "AdamOptimizer epsilon")
tf.flags.DEFINE_float("dropout_rate", "0.2", "dropout_rate")
# Momentum Optimizer will use
tf.flags.DEFINE_float("nesterov_momentum", "0.9", "nesterov_momentum")
tf.flags.DEFINE_float("weight_decay", "1e-4", "weight_decay")
# Label & batch_size
tf.flags.DEFINE_integer("batch_size", "100", "batch size for training")
tf.flags.DEFINE_integer("total_epochs", "50", "total number of epochs for training")
tf.flags.DEFINE_string('mode', "train", "Mode train/ test-dev/ test")


def main(args=None):
    print(args)
    tf.reset_default_graph()
    """
    Build Graph
    """
    with tf.Graph().as_default():
        """
        Dataset parser
        """
        FLAGS.network_name = args[0].split('/')[-1].split('.')[0]
        FLAGS.logs_dir = './logs_' + FLAGS.network_name
        parser = MNISTParser(dataset_dir='./dataset/coco_stuff', flags=FLAGS)
        mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
        """
        Input (TFRecord and Place_holder)
        """
        with tf.name_scope(name='Input'):
            x = tf.placeholder(tf.float32, shape=[None, 784])
            label = tf.placeholder(tf.float32, shape=[None, 10])
            training_flag = tf.placeholder(tf.bool)
            learning_rate = tf.placeholder(tf.float32, name='learning_rate')
            dropout_rate = tf.placeholder(tf.float32, name='dropout_rate')
            batch_images = tf.reshape(x, [-1, 28, 28, 1])
        """
        Network (Computes predictions from the inference model)
        """
        with tf.name_scope(name='Network'):
            with tf.variable_scope(FLAGS.network_name, reuse=False):
                logits = DenseNet(x=batch_images, nb_blocks=FLAGS.nb_block, filters=FLAGS.growth_k,
                                  training=training_flag, dropout_rate=dropout_rate, class_num=FLAGS.class_num).model
        """
        Optimize (Calculate loss to back propagate network)
        """
        with tf.name_scope(name='Optimize'):
            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=label, logits=logits))
            global_step = tf.Variable(0, trainable=False)
            epoch_learning_rate = tf.Variable(FLAGS.init_learning_rate, trainable=False)
            trainable_var = tf.get_collection(key=tf.GraphKeys.TRAINABLE_VARIABLES, scope=FLAGS.network_name)
            slim.model_analyzer.analyze_vars(trainable_var, print_info=True)
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                train_op = tf.train.AdamOptimizer(learning_rate=learning_rate, epsilon=FLAGS.epsilon).minimize(
                    loss=loss, global_step=global_step, var_list=trainable_var)
        """
        Other Variables 
        """
        with tf.name_scope(name='Others'):
            correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(label, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            summary_loss = tf.summary.scalar('loss', loss)
            summary_accuracy = tf.summary.scalar('accuracy', accuracy)
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
                ckpt = tf.train.get_checkpoint_state(parser.checkpoint_dir)
                if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
                    print("Model restored: {}".format(ckpt.model_checkpoint_path))
                    saver.restore(sess, ckpt.model_checkpoint_path)
                else:
                    print("No Model found.")
                    sess.run(init_op)
                # Tensor board
                merged = tf.summary.merge(inputs=[summary_loss, summary_accuracy], name='training')
                writer = tf.summary.FileWriter(parser.logs_dir, sess.graph)

            """
            Training Mode
            """
            if FLAGS.mode == 'train':
                for epoch in range(FLAGS.total_epochs):
                    # if epoch == (FLAGS.total_epochs * 0.5) or epoch == (FLAGS.total_epochs * 0.75):
                    #     epoch_learning_rate = epoch_learning_rate / 10

                    total_batch = mnist.train.num_examples // FLAGS.batch_size

                    for step in range(total_batch):
                        batch_x, batch_y = mnist.train.next_batch(FLAGS.batch_size)

                        train_feed_dict = {
                            x: batch_x,
                            label: batch_y,
                            learning_rate: epoch_learning_rate,
                            dropout_rate: FLAGS.dropout_rate,
                            training_flag: True
                        }

                        _, loss_sess = sess.run([train_op, loss], feed_dict=train_feed_dict)

                        if step % 100 == 0:
                            global_step += 100
                            train_summary, train_accuracy = sess.run([merged, accuracy], feed_dict=train_feed_dict)
                            # accuracy.eval(feed_dict=feed_dict)
                            print("Step:", step, "Loss:", loss_sess, "Training accuracy:", train_accuracy)
                            writer.add_summary(train_summary, global_step=epoch)

                    test_feed_dict = {
                        x: mnist.test.images,
                        label: mnist.test.labels,
                        learning_rate: epoch_learning_rate,
                        dropout_rate: 0.0,
                        training_flag: False
                    }

                    accuracy_rates = sess.run(accuracy, feed_dict=test_feed_dict)
                    print('Epoch:', '%04d' % (epoch + 1), '/ Accuracy =', accuracy_rates)
                    # writer.add_summary(test_summary, global_step=epoch)

                saver.save(sess=sess, save_path='./model/dense.ckpt')

if __name__ == "__main__":
    tf.app.run()
