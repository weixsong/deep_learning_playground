from ops import *
from utils import *
import os
import time
import argparse
import numpy as np
import tensorflow as tf
from glob import glob
from scipy.misc import imsave
from random import randint
import data.cifar10_data as cifar10_data
import data.imagenet_data as imagenet_data


# -----------------------------------------------------------------------------
parser = argparse.ArgumentParser()
# data I/O
parser.add_argument('-i', '--data_dir', type=str,
                    default='temp/data', help='Location for the dataset')
parser.add_argument('-d', '--data_set', type=str,
                    default='cifar', help='Can be either cifar|imagenet')
parser.add_argument('-lr', '--learning_rate', type=float,
                    default=0.001, help='Base learning rate')
parser.add_argument('-bs', '--batch_size', type=int, default=64,
                    help='Batch size during training per GPU')
parser.add_argument('-o', '--save_dir', type=str, default='./model',
                    help='Location for parameter checkpoints and samples')
parser.add_argument('-s', '--seed', type=int, default=8999,
                    help='Random seed to use')
parser.add_argument('-K', '--discriminator_steps', type=int, default=1,
                    help='discriminator update steps, default is 1')
parser.add_argument('-x', '--max_epochs', type=int,
                    default=50, help='How many epochs to run in total?')
parser.add_argument('-e', '--eval_steps', type=int, default=200,
                    help='Every how many step to eval model and save model')


args = parser.parse_args()


def make_discriminator(image, args, cifar=True, reuse=False):
    if reuse:
        tf.get_variable_scope().reuse_variables()

    if cifar:
        h0 = lrelu(conv2d(image, 3, df_dim, name='d_h0_conv')) #16x16x64
        h1 = lrelu(d_bn1(conv2d(h0, df_dim, df_dim*2, name='d_h1_conv'))) #8x8x128
        h2 = lrelu(d_bn2(conv2d(h1, df_dim*2, df_dim*4, name='d_h2_conv'))) #4x4x256
        h4 = dense(tf.reshape(h2, [args.batch_size, -1]), 4*4*df_dim*4, 1, scope='d_h3_lin')
        return tf.nn.sigmoid(h4), h4
    else:
        h0 = lrelu(conv2d(image, 3, df_dim, name='d_h0_conv'))
        h1 = lrelu(d_bn1(conv2d(h0, 64, df_dim*2, name='d_h1_conv')))
        h2 = lrelu(d_bn2(conv2d(h1, 128, df_dim*4, name='d_h2_conv')))
        h3 = lrelu(d_bn3(conv2d(h2, 256, df_dim*8, name='d_h3_conv')))
        h4 = dense(tf.reshape(h3, [args.batch_size, -1]), 4*4*512, 1, scope='d_h3_lin')
        return tf.nn.sigmoid(h4), h4


def make_generator(z, args, cifar=True):
    if cifar:
        z2 = dense(z, z_dim, 4*4*gf_dim*4, scope='g_h0_lin')
        h0 = tf.nn.relu(g_bn0(tf.reshape(z2, [-1, 4, 4, gf_dim*4]))) # 4x4x256
        h1 = tf.nn.relu(g_bn1(conv_transpose(h0, [args.batch_size, 8, 8, gf_dim*2], "g_h1"))) #8x8x128
        h2 = tf.nn.relu(g_bn2(conv_transpose(h1, [args.batch_size, 16, 16, gf_dim*1], "g_h2"))) #16x16x64
        h4 = conv_transpose(h2, [args.batch_size, 32, 32, 3], "g_h4")
        return tf.nn.tanh(h4)
    else:
        z2 = dense(z, z_dim, gf_dim*8*4*4, scope='g_h0_lin')
        h0 = tf.nn.relu(g_bn0(tf.reshape(z2, [-1, 4, 4, gf_dim*8])))
        h1 = tf.nn.relu(g_bn1(conv_transpose(h0, [args.batch_size, 8, 8, gf_dim*4], "g_h1")))
        h2 = tf.nn.relu(g_bn2(conv_transpose(h1, [args.batch_size, 16, 16, gf_dim*2], "g_h2")))
        h3 = tf.nn.relu(g_bn3(conv_transpose(h2, [args.batch_size, 32, 32, gf_dim*1], "g_h3")))
        h4 = conv_transpose(h3, [args.batch_size, 64, 64, 3], "g_h4")
        return tf.nn.tanh(h4)


with tf.Session() as sess:
    # network params
    DataLoader = {'cifar': cifar10_data.DataLoader,
                  'imagenet': imagenet_data.DataLoader}[args.data_set]

    rng = np.random.RandomState(args.seed)
    train_data = DataLoader(args.data_dir, 'train', args.batch_size,
                            rng=rng, shuffle=True, return_labels=False)
    image_shape = train_data.get_observation_size()  # e.g. a tuple (32,32,3)

    z_dim = 100
    gf_dim = 64
    df_dim = 64
    is_cifar = False
    if args.data_set == 'cifar':
        gf_dim = 32
        df_dim = 32
        is_cifar = True

    with tf.variable_scope(tf.get_variable_scope(), reuse=False):
        # batch norm
        d_bn1 = batch_norm(name='d_bn1')
        d_bn2 = batch_norm(name='d_bn2')
        d_bn3 = batch_norm(name='d_bn3')
    
        g_bn0 = batch_norm(name='g_bn0')
        g_bn1 = batch_norm(name='g_bn1')
        g_bn2 = batch_norm(name='g_bn2')
        g_bn3 = batch_norm(name='g_bn3')
    
        # build model
        images = tf.placeholder(tf.float32, [args.batch_size] + list(image_shape), name="real_images") # placeholder for real image
        zin = tf.placeholder(tf.float32, [None, z_dim], name="z")                               # placeholder for noise input
        generator = make_generator(zin, args)
        print('create generator for noise input done!')
        discriminator_prob, discriminator_logit = make_discriminator(images, args, cifar=is_cifar)
        print('create discriminator for real image done!')
        
        # reuse variables
        tf.get_variable_scope().reuse_variables()
    
        # make discriminator that use output of generator as input
        discriminator_fake_prob, discriminator_fake_logit = make_discriminator(generator, args, cifar=is_cifar, reuse=True)
        print('create discriminator for fake image done!')

        # discriminator should assign correct label for both real image and generated image from generator
        discriminator_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=discriminator_logit, labels=tf.ones_like(discriminator_logit)))
        discriminator_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=discriminator_fake_logit, labels=tf.zeros_like(discriminator_fake_logit)))

        # generator should learn to output image seems like to real image
        generator_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=discriminator_fake_logit, labels=tf.ones_like(discriminator_fake_logit)))
        discriminator_loss = discriminator_loss_real + discriminator_loss_fake

    t_vars = tf.trainable_variables()
    d_vars = [var for var in t_vars if 'd_' in var.name]
    g_vars = [var for var in t_vars if 'g_' in var.name]

    # network optimizer
    discriminator_opt = tf.train.AdamOptimizer(args.learning_rate).minimize(discriminator_loss, var_list=d_vars)
    generator_opt = tf.train.AdamOptimizer(args.learning_rate).minimize(generator_loss, var_list=g_vars)
    tf.initialize_all_variables().run()

    saver = tf.train.Saver(max_to_keep=10)

    counter = 1
    start_time = time.time()

    # noise inpu
    display_z = np.random.uniform(-1, 1, [args.batch_size, z_dim]).astype(np.float32)

    # save real image
    real_images = train_data.next(n=64)
    real_images = np.array(real_images) / 127.0 - 1.
    if args.data_set == 'cifar':
        save_name = 'results/cifar_real.jpg'
    else:
        save_name = 'results/imagenet_real.jpg'

    imsave(save_name, merge(real_images, [8,8]))

    # reset dataset
    train_data.reset()
    
    # makedir
    intermidiate_image_path = "results/" + args.data_set
    if not os.path.exists(intermidiate_image_path):
        os.makedirs(intermidiate_image_path)
        
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    train = True
    steps = 0
    if train:
        for epoch in range(args.max_epochs):
            for d in train_data:
                steps += 1

                # normalize the training data
                batch_images = d / 127.0 - 1.

                # noise input for generator
                batch_z = np.random.uniform(-1, 1, [args.batch_size, z_dim]).astype(np.float32)

                if steps % 10 == 0:
                    d_loss_value, _ = sess.run([discriminator_loss, discriminator_opt],feed_dict={images: batch_images, zin: batch_z})
                    g_loss_value, _ = sess.run([generator_loss, generator_opt],feed_dict={zin: batch_z})
                else:
                    sess.run([discriminator_opt],feed_dict={images: batch_images, zin: batch_z})
                    sess.run([generator_opt],feed_dict={zin: batch_z})

                if steps % 10 == 0:
                     print("Epoch: [%2d] step = %s Generator Loss = %s, Discriminator Loss = %s " % (epoch, str(steps), d_loss_value, g_loss_value))

                if steps % args.eval_steps == 0:
                    print("evaluate model of step %d" % (steps,))
                    g_images = sess.run([generator], feed_dict={zin: display_z})
                    print(np.shape(g_images[0]))
                    imsave(intermidiate_image_path + '/' + str(counter) + ".jpg", merge(g_images[0],[8,8]))
                    # save model
                    checkpoint_path = os.path.join(args.save_dir, 'model.ckpt')
                    saver.save(sess, checkpoint_path, global_step=steps)

        print("training finished!")
    else:
        saver.restore(sess, tf.train.latest_checkpoint(os.getcwd()+"/training/"))
        batch_z = np.random.uniform(-1, 1, [1, z_dim]).astype(np.float32)
        batch_z = np.repeat(batch_z, batchsize, axis=0)
        for i in xrange(z_dim):
            edited = np.copy(batch_z)
            edited[:,i] = (np.arange(0.0, batchsize) / (batchsize/2)) - 1
            sdata = sess.run([G],feed_dict={ zin: edited })
            ims("results/imagenet/"+str(i)+".jpg",merge(sdata[0],[8,8]))
