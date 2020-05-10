import tensorflow as tf
import layer_def as ld
import BasicConvLSTMCell
import functools
import numpy as np

FLAGS = tf.app.flags.FLAGS

class GAN:
    
    def __init__(self, data, truth, lambd):
        self.data = data
        self.truth = truth
        self.lambd = lambd
        self.D_prediction
        self.D_loss
        self.D_loss_adv
        self.D_accuracy
        self.G_prediction
        self.G_loss
        self.G_l2
        self.D_optimize
        self.G_optimize

    def lazy_property(function):
        attribute = '_cache_' + function.__name__

        @property
        @functools.wraps(function)
        def decorator(self):
            if not hasattr(self, attribute):
                setattr(self, attribute, function(self))
            return getattr(self, attribute)

        return decorator

    @lazy_property
    def D_prediction(self):

        def init_weight(shape):
            return tf.Variable(tf.random_normal(shape, stddev=0.01))

        weights = {
            'conv1': init_weight([3, 3, 3, 1, 9]),
            'fc': init_weight([FLAGS.seq_length*32*32*9, 81]), # T x L x W x input features 
            'out': init_weight([81, 2])
        }
        biases = {
            'conv1': init_weight([9]),
            'fc': init_weight([81]),
            'out': init_weight([2])
        }

        # first layer... 3dconv, relu, maxpooling
        conv1 = tf.nn.conv3d(self.data,
                             weights['conv1'],
                             strides=[1, 1, 1, 1, 1],
                             padding='SAME')
        l1 = tf.nn.relu(conv1+biases['conv1'])
        l1 = tf.nn.dropout(l1, FLAGS.D_keep_prob)

        # reshape tensor for fully connected layer
        l2 = tf.reshape(l1, [-1, weights['fc'].get_shape().as_list()[0]])

        # last layer
        l3 = tf.nn.relu(tf.matmul(l2, weights['fc'])+biases['fc'])
        l3 = tf.nn.dropout(l3, FLAGS.D_keep_prob)

        output = tf.matmul(l3, weights['out']) + biases['out']

        return output

    @lazy_property
    def D_loss(self):
        # define loss operation
        return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                logits=self.D_prediction,
                labels=self.truth))

    @lazy_property
    def D_loss_adv(self):
        # define loss operation
        prob = tf.nn.softmax(self.D_prediction)
        prob_gen = prob[:,1]        
        return -tf.reduce_mean(tf.log(1-prob_gen+0.00001))

    @lazy_property
    def D_optimize(self):
        # define a training operation
        return tf.train.AdamOptimizer(FLAGS.D_learning_rate).minimize(self.D_loss)

    @lazy_property
    def D_accuracy(self):
        # return binary classification accuracy
        correct_pred = tf.equal(tf.argmax(self.D_prediction, 1),
                                tf.argmax(self.truth, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
        return accuracy

    # define the generator model
    @lazy_property
    def G_prediction(self):

        # configure the input-cell-output structure
        def conv_lstm_cell(inputs, hidden):

            # some convolutional layers before the convLSTM cell
            conv1 = ld.conv_layer(inputs, 3, 2, 9, "encode_1")
            conv2 = ld.conv_layer(conv1, 3, 1, 9, "encode_2")

            # take output from first conv layers as input to convLSTM cell
            with tf.variable_scope('conv_lstm', initializer=tf.random_uniform_initializer(-.01, 0.1)):
                cell = BasicConvLSTMCell.BasicConvLSTMCell([16, 16], [3, 3], 9)
                if hidden is None:
                    hidden = cell.zero_state(FLAGS.batch_size, tf.float32)
                cell_output, hidden = cell(conv2, hidden)

            # some convolutional layers after the convLSTM cell
            conv7 = ld.transpose_conv_layer(cell_output, 3, 1, 9, "decode_7")
            # the last convolutional layer will use linear activations
            x_1 = ld.transpose_conv_layer(conv7, 3, 2, 1, "decode_8", True)

            # return the output of the last conv layer, & the hidden cell state
            return x_1, hidden

        # make a template for variable reuse
        cell = tf.make_template('cell', conv_lstm_cell)

        # cell outputs will be stored here
        x_unwrap = []

        # initialize hidden state to None in first cell
        hidden = None

        # loop over each frame in the sequence, sending through convLSTM cells
        for i in xrange(FLAGS.seq_length-1):

            # look at true frames for the first 'seq_start' samples
            if i < FLAGS.seq_start:
                x_1, hidden = cell(self.data[:, i, :, :, :], hidden)

            # after 'seq_start' samples, begin making predictions and
            # propagating through LSTM network
            else:
                x_1, hidden = cell(x_1, hidden)

            # add outputs to list
            x_unwrap.append(x_1)

        # stack and reorder predictions
        x_unwrap = tf.stack(x_unwrap)
        x_unwrap = tf.transpose(x_unwrap, [1, 0, 2, 3, 4])

        # return the prediction tensor
        return tf.maximum(x_unwrap, 0)

    @lazy_property
    def G_loss(self):
        # define loss operation
        # this is the L2 loss between true and predicted frames...
        adv = self.D_loss_adv
        l2 = tf.nn.l2_loss(self.data[:, FLAGS.seq_start:, :, :, :] -
                           self.G_prediction[:, FLAGS.seq_start-1:, :, :, :])        
        #return l2
        #return adv
        return (l2 + adv*self.lambd)
        
    @lazy_property
    def G_l2(self):
        # define loss operation
        # this is the L2 loss between true and predicted frames...
        l2 = tf.nn.l2_loss(self.data[:, FLAGS.seq_start:, :, :, :] -
                           self.G_prediction[:, FLAGS.seq_start-1:, :, :, :])
        return l2

    @lazy_property
    def G_optimize(self):
        # define a training operation
        return tf.train.AdamOptimizer(FLAGS.G_learning_rate).minimize(self.G_loss)
