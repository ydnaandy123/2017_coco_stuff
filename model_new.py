from ops import *

class DenseNet:
    def __init__(self, x, nb_blocks, filters, training, dropout_rate, class_num):
        self.nb_blocks = nb_blocks
        self.filters = filters
        self.training = training
        self.dropout_rate = dropout_rate
        self.class_num = class_num
        self.model = self.dense_net(x)

    def bottleneck_layer(self, x, scope):
        # print(x)
        with tf.variable_scope(scope):
            x = batch_normalization(x, training=self.training, name='batch1')
            x = relu(x)
            x = conv_layer(x, filters=4 * self.filters, kernel=[1, 1], name='conv1')
            x = drop_out(x, rate=self.dropout_rate, training=self.training)

            x = batch_normalization(x, training=self.training, name='batch2')
            x = relu(x)
            x = conv_layer(x, filters=self.filters, kernel=[3, 3], name='conv2')
            x = drop_out(x, rate=self.dropout_rate, training=self.training)
            # print(x)
            return x

    def transition_layer(self, x, scope):
        with tf.variable_scope(scope):
            x = batch_normalization(x, training=self.training, name='batch1')
            x = relu(x)
            x = conv_layer(x, filters=self.filters, kernel=[1, 1], name='conv1')
            x = drop_out(x, rate=self.dropout_rate, training=self.training)
            x = average_pooling(x)

            return x

    def dense_block(self, input_x, nb_layers, layer_name):
        with tf.variable_scope(layer_name):
            layers_concat = list()
            layers_concat.append(input_x)

            x = self.bottleneck_layer(input_x, scope=layer_name + '_bottleN_' + str(0))

            layers_concat.append(x)

            for i in range(nb_layers - 1):
                x = concatenation(layers_concat)
                x = self.bottleneck_layer(x, scope=layer_name + '_bottleN_' + str(i + 1))
                layers_concat.append(x)

            return x

    def dense_net(self, input_x):
        x = conv_layer(input_x, filters=2 * self.filters, kernel=[7, 7], stride=2, name='conv0')
        x = max_pooling(x, pool_size_h=3, pool_size_w=3, stride_h=2, stride_w=2,)

        for i in range(self.nb_blocks):
            # 6 -> 12 -> 48
            x = self.dense_block(input_x=x, nb_layers=4, layer_name='dense_'+str(i))
            x = self.transition_layer(x, scope='trans_'+str(i))

        x = self.dense_block(input_x=x, nb_layers=32, layer_name='dense_final')

        x = batch_normalization(x, training=self.training, name='linear_batch')
        x = relu(x)
        x = global_average_pooling(x)
        x = flatten_fixed(x)
        x = linear(x=x, units=self.class_num)

        return x
