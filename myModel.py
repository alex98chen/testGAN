import tensorflow as tf
import numpy as np
import random
import sys

dist = tf.contrib.distributions

def genRealData(batch_size = 128, n_mixture = 8, std = 0.01, radius = 1.0):
    thetas = np.linspace(0, 2 * np.pi - 2 * np.pi / n_mixture, n_mixture)
    centers = []
    for i in range(0, len(thetas)):
        centers.append([radius * np.sin(thetas[i]), radius * np.cos(thetas[i])])
    samples = []
    for c in centers:
        samples.extend(np.random.normal(loc=c, scale=std, size=[8192, 2]).tolist())
    random.shuffle(samples)
    #tensor = tf.convert_to_tensor(samples)
    #tensor.reshape([-1, 2, 1])
    return samples



def discriminator(inputs, reuse=False):
    with tf.variable_scope("discriminator", reuse = reuse):

        input_layer = tf.reshape(inputs, [-1, 2, 1])

        # First convolutional and pool layers
        # This finds 32 different 5 point features
        d_w1 = tf.get_variable('d_w1', [5, 1, 32], initializer=tf.truncated_normal_initializer(stddev=0.02))
        d_b1 = tf.get_variable('d_b1', [32], initializer=tf.constant_initializer(0))
        d1 = tf.nn.conv1d(input=input_layer, filter=d_w1, stride = 1, padding='SAME')
        d1 = d1 + d_b1
        d1 = tf.nn.relu(d1)

        # Second convolutional and pool layers
        # This finds 64 different 5 point features
        d_w2 = tf.get_variable('d_w2', [5, 32, 64], initializer=tf.truncated_normal_initializer(stddev=0.02))
        d_b2 = tf.get_variable('d_b2', [64], initializer=tf.constant_initializer(0))
        d2 = tf.nn.conv1d(input=d1, filter=d_w2, stride=1, padding='SAME')
        d2 = d2 + d_b2
        d2 = tf.nn.relu(d2)

        # First fully connected layer
        d_w3 = tf.get_variable('d_w3', [64, 1024], initializer=tf.truncated_normal_initializer(stddev=0.02))
        d_b3 = tf.get_variable('d_b3', [1024], initializer=tf.constant_initializer(0))
        d3 = tf.reshape(d2, [-1, 64])
        d3 = tf.matmul(d3, d_w3)
        d3 = d3 + d_b3
        d3 = tf.nn.relu(d3)

        # Second fully connected layer
        d_w4 = tf.get_variable('d_w4', [1024, 1], initializer=tf.truncated_normal_initializer(stddev=0.02))
        d_b4 = tf.get_variable('d_b4', [1], initializer=tf.constant_initializer(0))
        d4 = tf.matmul(d3, d_w4) + d_b4

        # d4 contains unscaled values
        return d4



def generator(z, batch_size = 128, layer_sizes= [1024, 512, 256, 128, 2]):
    with tf.variable_scope("generator"):
            reshape = tf.layers.dense(z, 4 * layer_sizes[0])
            reshape = tf.reshape(reshape, [-1, 4, layer_sizes[0]])
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





def main(argv):

    if argv[1] == "train":
        train_inputs = genRealData()


if __name__ == '__main__' :
    main(sys.argv)