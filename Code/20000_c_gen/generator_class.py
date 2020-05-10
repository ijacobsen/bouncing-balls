import tensorflow as tf
import layer_def as ld
import BasicConvLSTMCell
import functools

FLAGS = tf.app.flags.FLAGS


def lazy_property(function):
    attribute = '_cache_' + function.__name__

    @property
    @functools.wraps(function)
    def decorator(self):
        if not hasattr(self, attribute):
            setattr(self, attribute, function(self))
        return getattr(self, attribute)

    return decorator


# define the generator model
class generator:

    def __init__(self, data):
        self.data = data
        self.prediction
        self.optimize
        self.loss

    @lazy_property
    def prediction(self):

        # configure the input-cell-output structure
        def conv_lstm_cell(inputs, hidden):

            # some convolutional layers before the convLSTM cell
            conv1 = ld.conv_layer(inputs, 3, 2, 4, "encode_1")
            conv2 = ld.conv_layer(conv1, 3, 1, 8, "encode_2")
            conv3 = ld.conv_layer(conv2, 3, 1, 16, "encode_3")

            # take output from first conv layers as input to convLSTM cell
            with tf.variable_scope('conv_lstm', initializer=tf.random_uniform_initializer(-.01, 0.1)):
                cell = BasicConvLSTMCell.BasicConvLSTMCell([16, 16], [3, 3], 16)
                if hidden is None:
                    hidden = cell.zero_state(FLAGS.batch_size, tf.float32)
                cell_output, hidden = cell(conv3, hidden)

            # some convolutional layers after the convLSTM cell
            conv5 = ld.transpose_conv_layer(cell_output, 1, 1, 16, "decode_5")
            conv6 = ld.transpose_conv_layer(conv5, 3, 1, 8, "decode_6")
            conv7 = ld.transpose_conv_layer(conv6, 3, 1, 4, "decode_7")

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
        return x_unwrap

    @lazy_property
    def loss(self):
        # define loss operation
        # this is the L2 loss between true and predicted frames...
        return tf.nn.l2_loss(self.data[:, FLAGS.seq_start:, :, :, :] -
                             self.prediction[:, FLAGS.seq_start-1:, :, :, :])

    @lazy_property
    def optimize(self):
        # define a training operation
        return tf.train.AdamOptimizer(FLAGS.learning_rate).minimize(self.loss)
