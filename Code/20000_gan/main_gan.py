from GAN_class import *
import data_handler as dh
import tensorflow as tf
import numpy as np
import skvideo.io
import scipy.io as sio
from matplotlib import pyplot as plt

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('train_dir', './trained_model',
                           """ directory to store trained model""")
tf.app.flags.DEFINE_string('mode', 'gen_video',
                           """ adv_train, or gen_video """)
tf.app.flags.DEFINE_integer('seq_length', 10,
                            """size of hidden layer""")
tf.app.flags.DEFINE_integer('seq_start', 5,
                            """ start of seq generation""")
tf.app.flags.DEFINE_integer('max_step', 20000,
                            """maximum number of steps""")
tf.app.flags.DEFINE_integer('k', 7,
                            """how many steps to train G for each D""")
tf.app.flags.DEFINE_integer('pretrain_G', 10,
                            """we must pretrain G and D""")
tf.app.flags.DEFINE_integer('pretrain_D', 50,
                            """we must pretrain G and D""")
tf.app.flags.DEFINE_float('lambd', 800,
                          """adversarial loss multiplier""")
tf.app.flags.DEFINE_float('keep_prob', .8,
                          """for dropout""")
tf.app.flags.DEFINE_float('D_keep_prob', .8,
                          """for dropout""")
tf.app.flags.DEFINE_float('G_learning_rate', .01,
                          """learning rate for G""")
tf.app.flags.DEFINE_float('D_learning_rate', .001,
                          """learning rate for D""")
tf.app.flags.DEFINE_integer('batch_size', 16,
                            """batch size for training""")
tf.app.flags.DEFINE_float('weight_init', .1,
                          """weight init for fully connected layers""")
tf.app.flags.DEFINE_integer('gen_len', 70,
                            """number of frames to generate in gen_video mode""")
tf.app.flags.DEFINE_bool('friction', False,
                         """whether there is friction in the system""")
tf.app.flags.DEFINE_integer('num_balls', 2,
                            """num of balls in the simulation""")     

# configure input parameters... stored as flags
if tf.gfile.Exists(FLAGS.train_dir):
    tf.gfile.DeleteRecursively(FLAGS.train_dir)
tf.gfile.MakeDirs(FLAGS.train_dir)

######### BUILD GRAPHS #########
# make input tensor, and wrap it with dropout
x = tf.placeholder(tf.float32, [None, FLAGS.seq_length, 32, 32, 1])
y = tf.placeholder(tf.int8, [None, 2])
lambd = tf.placeholder(tf.float32)

# create D and G
model = GAN(x, y, lambd)

# define an initialization operation
init_op = tf.global_variables_initializer()

# create saving operation
saver = tf.train.Saver()

# run operations on the graph
with tf.Session() as sess:

    print('building network')
    sess.run(init_op)

    print('creating output file')

    # if we're not using a trained network to generate a video, then
    # independently pretrain G and D
    if FLAGS.mode != 'gen_video':

        print('training G')

        # targets for D, one hot vectors
        targ = np.ones((FLAGS.batch_size, 2)) 
        
        # first we'll pretrain G
        for i in range(FLAGS.pretrain_G):
            # generate a batch of training data
            data = dh.generate_bouncing_ball_sample(FLAGS.batch_size,
                                                    FLAGS.seq_length,
                                                    [32, 32, 1],
                                                    FLAGS.num_balls)

            # run training and loss operations on the graph
            _, loss_G = sess.run([model.G_optimize,
                                  model.G_loss],
                                 feed_dict={x: data,
                                            y: targ,
                                            lambd: 0})
            print(loss_G)
        
        # then we'll pretrain D
        print('training D')

        for i in range(FLAGS.pretrain_D):
            
            # prior data
            data = dh.generate_bouncing_ball_sample(FLAGS.batch_size,
                                                    FLAGS.seq_length,
                                                    [32, 32, 1],
                                                    FLAGS.num_balls)

            # on even iters sample batch of examples (z) from G
            if i % 2 == 0:
                # prediction from network on prior
                pred = sess.run([model.G_prediction],
                                feed_dict={x: data,
                                           y: targ,
                                           lambd: 0})
        
                # clean up the prediction and cut out the last few frames
                pred = np.maximum(pred[0], 0)
                gen_frames = FLAGS.seq_length - FLAGS.seq_start
                gen_data = pred[:, -gen_frames:, :, :, :]
    
                # true data to use
                true_data = data[:, :FLAGS.seq_start, :, :, :]
    
                # stitch true frames with generated frames
                data = np.concatenate((true_data, gen_data), axis=1)
                targ[:, :] = [0, 1]
                print('gen:')

            # on odd iters sample batch of examples (x) from true source
            else:
                print('truth:')
                targ[:, :] = [1, 0]

            # train D
            _, D_acc = sess.run([model.D_optimize,
                                         model.D_accuracy],
                                        feed_dict={x: data,
                                                   y: targ,
                                                   lambd: 1})
            #print()
            print('D Accuracy: {}'.format(D_acc))

        print('pretraining complete... D Accuracy: {}, G loss: {} \n'.format(D_acc, loss_G))

    if FLAGS.mode == 'adv_train':

    	f = open('output.txt', 'w')
        # targets for D... one hot vectors
        targ = np.ones((FLAGS.batch_size, 2))

        # iterate the number of training steps
        for step in xrange(FLAGS.max_step):
            converg_sum = 0
            sum_acc = 0
            ave_acc = 0

            # for k steps we train D
            for i in range(FLAGS.k):

                # prior data
                data = dh.generate_bouncing_ball_sample(FLAGS.batch_size,
                                                        FLAGS.seq_length,
                                                        [32, 32, 1],
                                                        FLAGS.num_balls)

                # on even iters sample batch of examples (z) from G
                if i % 2 == 0:

                    # prediction from network on prior
                    pred = sess.run([model.G_prediction],
                                    feed_dict={x: data,
                                               y: targ,
                                               lambd: 0})

                    # clean up the prediction and cut out the last few frames
                    pred = np.maximum(pred[0], 0)
                    gen_frames = FLAGS.seq_length - FLAGS.seq_start
                    gen_data = pred[:, -gen_frames:, :, :, :]

                    # true data to use
                    true_data = data[:, :FLAGS.seq_start, :, :, :]

                    # stitch true frames with generated frames
                    data = np.concatenate((true_data, gen_data), axis=1)
                    targ[:, :] = [0, 1]

                # on odd iters sample batch of examples (x) from true source
                else:
                    targ[:, :] = [1, 0]

                # train D
                _, D_acc = sess.run([model.D_optimize,
                                     model.D_accuracy],
                                    feed_dict={x: data,
                                               y: targ,
                                               lambd: 0})

                sum_acc = sum_acc + D_acc
                ave_acc = sum_acc/(i+1)
                
                f.write('step: {}, k: {}, D acc: {:.2f}, ave_acc: {:.2f} \n'.format(step, i,
                                                                                    D_acc, ave_acc))
                print('step: {}, k: {}, D acc: {:.2f}, ave acc: {:.2f} \n'.format(step, i,
                                                                                  D_acc, ave_acc))

                
                converg_sum = np.floor(D_acc)*(D_acc + converg_sum)
                
                if converg_sum > 5:
                    converg_sum = 0
                    print('discriminator converged')
                    break

            # generate a batch of training data
            data = dh.generate_bouncing_ball_sample(FLAGS.batch_size,
                                                    FLAGS.seq_length,
                                                    [32, 32, 1],
                                                    FLAGS.num_balls)

            # run training and loss operations on the graph
            targ[:, :] = [0, 1]
            _, loss_G, l2, ladv = sess.run([model.G_optimize,
                                            model.G_loss,
                                            model.G_l2,
                                            model.D_loss_adv],
                                           feed_dict={x: data,
                                                      y: targ,
                                                      lambd: FLAGS.lambd})

            f.write('step: {}, G loss: {}, l2 loss: {}, adv loss: {} \n'.format(step,
                                                                                loss_G,
                                                                                l2,
                                                                                FLAGS.lambd*ladv))
            print('step: {}, G loss: {}, l2 loss: {}, adv loss: {} \n'.format(step,
                                                                              loss_G,
                                                                              l2,
                                                                              FLAGS.lambd*ladv))
            if step % 100 == 0:
                save_path = saver.save(sess, FLAGS.train_dir)
                print('model saved in file {}'.format(save_path))
        f.close()

    if FLAGS.mode == 'gen_video':
        saver.restore(sess, FLAGS.train_dir)
        print('model restored')
        
        # targets for D... one hot vectors... values don't matter here
        targ = np.ones((FLAGS.batch_size, 2))

        data = dh.generate_bouncing_ball_sample(FLAGS.batch_size,
                                                FLAGS.seq_length,
                                                [32, 32, 1],
                                                FLAGS.num_balls)

        # some video stuff
        video = np.zeros([FLAGS.gen_len, 32, 32, 1])

        # loop over generate length
        for i in range(FLAGS.gen_len):

            # predict the next 'seq_length - seq_start' frames
            pred = sess.run([model.G_prediction],
                            feed_dict={x: data,
                                       y: targ,
                                       lambd: 0})

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
        skvideo.io.vwrite('data.mp4', video, backend='libav')
        
