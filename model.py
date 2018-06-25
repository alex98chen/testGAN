#using sugyan tf-dcgan as outline

import tensorflow as tf
import numpy as np
import sys
import matplotlib.pyplot as plt
import random

ds = tf.contrib.distributions


def genGausSamples(batch_size = 128, n_mixture=8, std = 0.01, radius = 1.0):
    thetas = np.linspace(0, 2 * np.pi - 2 * np.pi / n_mixture, n_mixture)
    centers = []
    for i in range(0, len(thetas)):
        centers.append([radius * np.sin(thetas[i]), radius * np.cos(thetas[i])])
    samples = []
    for c in centers:
        samples.extend(np.random.normal(loc=c, scale=std, size=[8192, 2]))
    for s in range(len(samples)):
        samples[s] = tf.convert_to_tensor(samples[s])
    random.shuffle(samples)
    #tensor = tf.convert_to_tensor(samples)
    #tensor.reshape([-1, 2, 1])
    return samples



class Generator:


    #figure out what s_size means
    #decide if brackets should hold 3 as original code did
    def __init__(self, outputSize = 2, layer_sizes = [1024, 512, 256, 128], s_size = 4):
        self.layer_sizes = layer_sizes + [outputSize]
        self.s_size = 4
        self.reuse = False

    def __call__(self, inputs, training = False):
        inputs = tf.convert_to_tensor(inputs)
        with tf.variable_scope('generator', reuse = self.reuse):
            #reshape the inputs
            reshape = tf.layers.dense(inputs, self.layer_sizes[0] * 2)
            reshape = tf.reshape(reshape, [-1, self.s_size, self.s_size, self.layer_sizes[0]])
            reshape = tf.nn.leaky_relu(tf.layers.batch_normalization(reshape, training = training) , name = 'reshape')

            #deconvolve 1
            deconv1 = tf.layers.conv2d_transpose(reshape, self.layer_sizes[1], [5, 5], strides = [2, 2], padding = 'SAME')
            deconv1 = tf.nn.leaky_relu(tf.layers.batch_normalization(deconv1, training = training), name = 'deconv1')

            deconv2 = tf.layers.conv2d_transpose(deconv1, self.layer_sizes[2], [5, 5], strides = [2, 2], padding = 'SAME')
            deconv2 = tf.nn.leaky_relu(tf.layers.batch_normalization(deconv2, training=training), name='deconv2')

            deconv3 = tf.layers.conv2d_transpose(deconv2, self.layer_sizes[3], [5, 5], strides = [2, 2], padding = 'SAME')
            deconv3 = tf.nn.leaky_relu(tf.layers.batch_normalization(deconv3, training = training), name = 'deconv3')

            deconv4 = tf.layers.conv2d_transpose(deconv3, self.layer_sizes[4], [5, 5], strides = [2, 2], padding = 'SAME')

            outputs = tf.tanh(deconv4, name = 'outputs')

        self.reuse = True
        self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope = 'generator')
        return outputs


#Should I use input_size??
class Discriminator:
    def __init__(self, inputSize = 2, layer_sizes = [64, 128, 256, 512]):
        self.layer_sizes = [inputSize] + layer_sizes
        self.reuse = False

    def __call__(self, inputs, training = False, name = ''):

        #changed this idk why
        #inputs = np.asarray(inputs)
        #print(inputs)
        inputs = tf.convert_to_tensor(inputs)
        print(inputs.get_shape())
        inputs = tf.reshape(inputs, [128, -1, 2])
        print(inputs.get_shape())
        print("HERE :)")
        with tf.variable_scope('discriminator', reuse = self.reuse):
            conv1 = tf.layers.conv1d(inputs, self.layer_sizes[1], 5, strides = 2, padding = 'SAME')
            conv1 = tf.nn.leaky_relu(tf.layers.batch_normalization(conv1, training=training), name = 'conv1')

            conv2 = tf.layers.conv1d(conv1, self.layer_sizes[2], 5, strides =2, padding = 'SAME')
            conv2 = tf.nn.leaky_relu(tf.layers.batch_normalization(conv2, training = training), name = 'conv2')

            conv3 = tf.layers.conv1d(conv2, self.layer_sizes[3], 5, strides = 2, padding = 'SAME')
            conv3 = tf.nn.leaky_relu(tf.layers.batch_normalization(conv3, training = training), name = 'conv3')

            conv4 = tf.layers.conv1d(conv3, self.layer_sizes[4], 5, strides = 2, padding = 'SAME')
            conv4 = tf.nn.leaky_relu(tf.layers.batch_normalization(conv4, training = training), name = 'conv4')

            batch_size = conv4.get_shape()[0].value
            reshape = tf. reshape(conv4, [batch_size, -1])
            outputs = tf.layers.dense(reshape, 128, name = 'outputs')

        self.reuse = True
        self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope = 'discrimninator')
        return outputs












class DCGAN:
    def __init__(self, batch_size = 128, s_size = 4, noise_dim = 100,
                 generator_depths = [1024, 512, 256, 128], discriminator_depths = [64, 128, 256, 512]):
        self.batch_size = batch_size;
        self.s_size = s_size;
        self.noise_dim = noise_dim
        self.gen = Generator(layer_sizes = generator_depths, s_size= self.s_size)
        self.dis = Discriminator(layer_sizes = discriminator_depths)
        self.noise = tf.random_uniform([self.batch_size, self.noise_dim], minval = -1, maxval = 1)

    def loss(self, traindata):
        generated = self.gen(self.noise, training = True)
        true_outputs = self.dis(traindata, training=True, name='true')
        fake_outputs = self.dis(generated, training = True, name = 'fake')

        tf.add_to_collection('generator_losses', tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels = tf.ones([self.batch_size], dtype = tf.int64), logits = fake_outputs
            )
        ))
        print("just did gen")

        tf.add_to_collection('discriminator_losses', tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels = tf.ones([self.batch_size], dtype=tf.int64), logits = true_outputs
            )
        ))
        print("just did disc")

        tf.add_to_collection('discriminator_losses', tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels = tf.zeros([self.batch_size], dtype = tf.int64), logits = fake_outputs
            )
        ))
        print("just did disc")

        return {self.gen: tf.add_n(tf.get_collection('generator_losses'), name = 'total_generator_loss'),
                self.dis: tf.add_n(tf.get_collection('disciminator_losses'), name = 'total_discriminator_loss')
                }

    def train(self, losses, learning_rate = 0.0002, beta1 = 0.5):
        g_opt = tf.train.AdamOptimizer(learning_rate = learning_rate, beta1 = beta1)
        d_opt = tf.train.AdamOptimizer(learning_rate = learning_rate, beta1 = beta1)
        g_opt_op = g_opt.minimize(losses[self.gen], var_list = self.gen.variables)
        d_opt_op = d_opt.minimize(losses[self.dis], var_list = self.dis_variables)
        with tf.control_dependencies([g_opt_op, d_opt_op]):
            return tf.no_op(name = 'train')

    def sample_images(self, inputs = None):
        if inputs is None:
            inputs = self.noise
        outputs = self.gen(inputs, training = True)
        outputs = [output for output in tf.split(outputs, self.batch_size, axis=0)]
        return outputs


def main(argv):
    dcgan = DCGAN()
    print("STARTING UP")

    if argv[1] == "train":
        train_inputs = genGausSamples()
        print("made samples")
        losses = dcgan.loss(train_inputs)
        print("made losses")
        train_op = dcgan.train(losses)
        print("Done setting up")

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            print('Running')
            for step in range(25000):
                print(step)
                _, g_loss_value, d_loss_value = sess.run([train_op, losses[dcgan.g],losses[dcgan.d]])

    if argv[1] == "generate":
        outputs = dcgan.sampe_images()
        with tf.Session() as sess:
            generated = []
            for i in range(100):
                generated.append(sess.run(outputs))
            x,y = generated.T
            plt.scatter(x, y)
            plt.show()



if __name__ == '__main__':
    main(sys.argv)