from generator_class import *
import numpy as np
import data_handler as dh
import tensorflow as tf
import numpy as np
import skvideo.io
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('band', 'coarse',
                           """ coarse or detail """)
tf.app.flags.DEFINE_integer('k', '1',
                           """ 1, 2, 3, 4, 5 (which db) """)
tf.app.flags.DEFINE_integer('m', '1',
                           """ number of levels (keep at 1) """)
tf.app.flags.DEFINE_string('train_dir', './trained_model_og',
                           """ directory to store trained model""")
tf.app.flags.DEFINE_string('mode', 'test',
                           """ train or test """)
tf.app.flags.DEFINE_integer('gen_len', 70,
                            """number of frames to generate in test mode""")
tf.app.flags.DEFINE_integer('seq_length', 10,
                            """size of hidden layer""")
tf.app.flags.DEFINE_integer('seq_start', 5,
                            """ start of seq generation""")
tf.app.flags.DEFINE_integer('max_step', 20000,
                            """max num of steps""")
tf.app.flags.DEFINE_float('keep_prob', .8,
                          """for dropout""")
tf.app.flags.DEFINE_float('learning_rate', .001,
                          """for dropout""")
tf.app.flags.DEFINE_integer('batch_size', 8,
                            """batch size for training""")
tf.app.flags.DEFINE_float('weight_init', .1,
                          """weight init for fully connected layers""")
tf.app.flags.DEFINE_bool('friction', False,
                         """whether there is friction in the system""")
tf.app.flags.DEFINE_integer('num_balls', 2,
                            """num of balls in the simulation""")

# configure input parameters... stored as flags
if tf.gfile.Exists(FLAGS.train_dir):
    tf.gfile.DeleteRecursively(FLAGS.train_dir)
tf.gfile.MakeDirs(FLAGS.train_dir)

# make input tensor, and wrap it with dropout
x = tf.placeholder(tf.float32, [None, FLAGS.seq_length, 32, 32, 1])
keep_prob = tf.placeholder("float")
x_dropout = tf.nn.dropout(x, keep_prob)

model = generator(x_dropout)

# define an initialization operation
init_op = tf.global_variables_initializer()

# create saving operation
saver = tf.train.Saver()

# run operations on the graph
with tf.Session() as sess:

    print('building network')
    sess.run(init_op)

    if FLAGS.mode == 'train':
	f = open('output.txt', 'w')
        loss = np.empty(FLAGS.max_step)
        # run specified number of training steps
        for step in xrange(FLAGS.max_step):

            # generate a batch of training data
            data = dh.generate_bouncing_ball_sample(FLAGS.batch_size,
                                            	  FLAGS.seq_length,
                                             	 [32, 32, 1],
                                             	 FLAGS.num_balls)

            # run training and loss operations on the graph
            _, loss_r = sess.run([model.optimize, model.loss],
                                 feed_dict={x: data,
                                            keep_prob: FLAGS.keep_prob})
            loss[step] = loss_r

	    f.write('step: {}, loss: {} \n'.format(step, loss_r))
            # print loss for troubleshooting... should be decreasing in trend
            if step % 100 == 0:
        	save_path = saver.save(sess, FLAGS.train_dir)
        	print('model saved in file {}'.format(save_path))

            assert not np.isnan(loss_r), 'Model diverged with loss = NaN'

	f.close()
        
        plt.figure(figsize=(8, 8))
        plt.plot(loss)
        plt.title('L2 Loss', fontsize=24)
        plt.xlabel('Iterations', fontsize=18)
        plt.ylabel('Loss', fontsize=18)
        plt.savefig('l2_loss_iter={}.png'.format(FLAGS.max_step))

    if FLAGS.mode == 'test':
        saver.restore(sess, FLAGS.train_dir)
        print('model restored')

        data = dh.generate_bouncing_ball_sample(FLAGS.batch_size,
                                         	FLAGS.seq_length,
                                          	[32, 32, 1],
                                          	FLAGS.num_balls)
        # some video stuff
        video = np.zeros([FLAGS.gen_len, 32, 32, 1])
        
        # loop over generate length
        for i in range(FLAGS.gen_len):

            # predict the next 'seq_length - seq_start' frames
            pred = sess.run([model.prediction], feed_dict={x: data,
                            keep_prob: 1.0})

            # we only want to use the prediction forecasted at time t+1
            pred_one = pred[0][0, FLAGS.seq_start-1, :, :, :]

            # update data
            data = np.roll(data, -1, axis=1)
            data[0, FLAGS.seq_start-1, :, :, :] = pred_one

            # write video frame
            frame = np.uint8(np.maximum(data[0, FLAGS.seq_start-1, :, :, :],
                                        0) * 255)
            video[i, :, :, :] = frame
            
        skvideo.io.vwrite('gen_video.mp4', video, backend='libav')
