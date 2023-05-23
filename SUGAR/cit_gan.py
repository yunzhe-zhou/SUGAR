import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import os
import time
import math
from datetime import datetime
import logging
from sklearn.metrics.pairwise import rbf_kernel
from scipy.stats import ks_2samp
from scipy.stats import wilcoxon
import random
from scipy import stats
import time
from collections import defaultdict
import warnings
from scipy.stats import rankdata

logging.getLogger('tensorflow').disabled = True
print("TensorFlow version:", tf.__version__)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.keras.backend.set_floatx('float32')
total_start_time = time.time()


# =================================================================================================================
# This defines the class of GAN structures
# =================================================================================================================


class WGanGenerator(tf.keras.Model):
    '''
    class for WGAN generator
    
    Parameters
    ----------
    n_samples: the total sample size
    z_dims: the dimension of the conditional component
    h_dims: the dimension of the hidden units
    v_dims: the dimension of the input noise
    batch_size: batch size for training
    '''
    def __init__(self, n_samples, z_dims, h_dims, v_dims, batch_size):
        super(WGanGenerator, self).__init__()
        # define the sample size, number of hidden units and the batch size for training
        self.n_samples = n_samples
        self.hidden_dims = h_dims
        self.batch_size = batch_size
        
        # define the structure of the neural network
        self.input_dim = z_dims + v_dims
        self.input_shape1 = [self.input_dim, self.hidden_dims]
        self.input_shape2 = [self.hidden_dims, self.hidden_dims]
        self.input_shape3 = [self.hidden_dims, 1]
        
        # initialize the weights of the neural network
        self.w1 = self.xavier_var_creator(self.input_shape1)
        self.b1 = tf.Variable(tf.zeros(self.input_shape1[1], tf.float64))

        self.w2 = self.xavier_var_creator(self.input_shape2)
        self.b2 = tf.Variable(tf.zeros(self.input_shape2[1], tf.float64))

        self.w3 = self.xavier_var_creator(self.input_shape3)
        self.b3 = tf.Variable(tf.zeros(self.input_shape3[1], tf.float64))

    def xavier_var_creator(self, input_shape):
        # function used for initialization
        xavier_stddev = 1.0 / tf.sqrt(input_shape[0] / 2.0)
        init = tf.random.normal(shape=input_shape, mean=0.0, stddev=xavier_stddev)
        init = tf.cast(init, tf.float64)
        var = tf.Variable(init, shape=tf.TensorShape(input_shape), trainable=True)
        return var

    def call(self, inputs, training=None, mask=None):
        # forward propogation of the neural network
        # inputs are concatenations of z and v
        z = tf.reshape(tensor=inputs, shape=[-1, self.input_dim])
        h1 = tf.nn.relu(tf.matmul(z, self.w1) + self.b1)
#        h2 = tf.nn.relu(tf.matmul(h1, self.w2) + self.b2)
        out = tf.math.sigmoid(tf.matmul(h1, self.w3) + self.b3)
        return out


class WGanDiscriminator(tf.keras.Model):
    '''
    class for WGAN discriminator
    
    Parameters
    ----------
    n_samples: the total sample size
    z_dims: the dimension of the conditional component
    h_dims: the dimension of the hidden units
    x_dims: the dimension of the data input
    batch_size: batch size for training
    '''
    def __init__(self, n_samples, z_dims, h_dims, x_dims, batch_size):
        super(WGanDiscriminator, self).__init__()
        # define the sample size, number of hidden units and the batch size for training
        self.n_samples = n_samples
        self.hidden_dims = h_dims
        self.batch_size = batch_size
        
        # define the structure of the neural network
        self.input_dim = z_dims + x_dims
        # self.input_dim = x_dims
        self.input_shape1 = [self.input_dim, self.hidden_dims]
        self.input_shape2 = [self.hidden_dims, self.hidden_dims]
        self.input_shape3 = [self.hidden_dims, 1]
        
        # initialize the weights of the neural network
        self.w1 = self.xavier_var_creator(self.input_shape1)
        self.b1 = tf.Variable(tf.zeros(self.input_shape1[1], tf.float64))

        self.w2 = self.xavier_var_creator(self.input_shape2)
        self.b2 = tf.Variable(tf.zeros(self.input_shape2[1], tf.float64))

        self.w3 = self.xavier_var_creator(self.input_shape3)
        self.b3 = tf.Variable(tf.zeros(self.input_shape3[1], tf.float64))

    def xavier_var_creator(self, input_shape):
        # function used for initialization
        xavier_stddev = 1.0 / tf.sqrt(input_shape[0] / 2.0)
        init = tf.random.normal(shape=input_shape, mean=0.0, stddev=xavier_stddev)
        init = tf.cast(init, tf.float64)
        var = tf.Variable(init, shape=tf.TensorShape(input_shape), trainable=True)
        return var

    def call(self, inputs, training=None, mask=None):
        # forward propogation of the neural network
        # inputs are concatenations of z and v
        z = tf.reshape(tensor=inputs, shape=[self.batch_size, -1])
        z = tf.cast(z, tf.float64)
        h1 = tf.nn.relu(tf.matmul(z, self.w1) + self.b1)
        out = tf.matmul(h1, self.w3) + self.b3
#         h2 = tf.nn.sigmoid(tf.matmul(h1, self.w2) + self.b2)
#         out = tf.nn.sigmoid(tf.matmul(h1, self.w3) + self.b3)
        return out


class Discriminator(tf.keras.layers.Layer):
    '''
    class to construct a function that represents the characteristic function
    
    Parameters
    ----------
    size: the total sample size
    z_dims: the dimension of the conditional component
    x_dims: the dimension of the data input
    '''

    # def __init__(self, size, x_dims, z_dims):
    def __init__(self, size, x_dims, z_dims, output_activation='linear'):
        super(Discriminator, self).__init__()
        # define the sample size, number of hidden units
        self.n_samples = size
        self.hidden_dims = 1
        
        # define the structure of the neural network
        self.input_dim = z_dims + x_dims
        self.z_dims = z_dims
        self.x_dims = x_dims
        self.input_shape1x = [self.x_dims, self.hidden_dims]
        self.input_shape1z = [self.z_dims, self.hidden_dims]
        self.input_shape1 = [self.input_dim, self.hidden_dims]
        self.input_shape2 = [self.hidden_dims, 1]
        
        # initialize the weights of the neural network
        self.w1x = self.xavier_var_creator(self.input_shape1x)
        self.w1z = self.xavier_var_creator(self.input_shape1z)
        # self.w1z = tf.Variable(tf.zeros(self.input_shape1z, tf.float64))
        self.w1 = tf.concat([self.w1x, self.w1z], axis=0)
        self.b1 = tf.squeeze(self.xavier_var_creator([self.x_dims, 1]))
        # self.b1 = tf.Variable(tf.zeros(self.input_shape1[1], tf.float64))

        self.w2 = self.xavier_var_creator(self.input_shape2)
        self.b2 = tf.Variable(tf.zeros(self.input_shape2[1], tf.float64))

    def xavier_var_creator(self, input_shape):
        # function used for initialization
        xavier_stddev = tf.sqrt(2.0 / (input_shape[0]))
        init = tf.random.normal(shape=input_shape, mean=0.0, stddev=xavier_stddev)
        init = tf.cast(init, tf.float64)
        var = tf.Variable(init, shape=tf.TensorShape(input_shape), trainable=True)
        return var

    def call(self, x, z):
        # forward propogation of the neural network
        # inputs are concatenations of z and v
        # z = tf.reshape(tensor=inputs, shape=[-1, self.input_dim])
        x = tf.reshape(tensor=x, shape=[-1, self.x_dims])
        z = tf.reshape(tensor=z, shape=[-1, self.z_dims])

        z_h1 = tf.nn.tanh(tf.matmul(z, self.w1z) + self.b1)
        x_h1 = tf.nn.tanh(tf.matmul(x, self.w1x) + self.b1)

        # h1 = tf.nn.sigmoid(tf.matmul(z, self.w1z) + tf.matmul(x, self.w1x) + self.b1)
        # h1 = tf.nn.sigmoid(tf.matmul(z, self.w1z) + tf.matmul(x, self.w1x) + self.b1)
        # out = tf.nn.sigmoid(tf.matmul(h1, self.w2))
        # out = tf.matmul(h1, self.w2) + self.b2
        out = z_h1 * x_h1
        return out


class MINEDiscriminator(tf.keras.layers.Layer):
    '''
    class for MINE discriminator
    '''

    def __init__(self, in_dims, output_activation='linear'):
        super(MINEDiscriminator, self).__init__()
        self.output_activation = output_activation
        self.input_dim = in_dims
        
        # initialize the weights of the neural network
        self.w1a = self.xavier_var_creator()
        self.w1b = self.xavier_var_creator()
        self.b1 = tf.Variable(tf.zeros([self.input_dim, ], tf.float64))

        self.w2a = self.xavier_var_creator()
        self.w2b = self.xavier_var_creator()
        self.b2 = tf.Variable(tf.zeros([self.input_dim, ], tf.float64))

        self.w3 = self.xavier_var_creator()
        self.b3 = tf.Variable(tf.zeros([self.input_dim, ], tf.float64))

    def xavier_var_creator(self):
        # function used for initialization
        xavier_stddev = 1.0 / tf.sqrt(self.input_dim / 2.0)
        init = tf.random.normal(shape=[self.input_dim, ], mean=0.0, stddev=xavier_stddev)
        init = tf.cast(init, tf.float64)
        var = tf.Variable(init, shape=tf.TensorShape(self.input_dim, ), trainable=True)
        return var

    def mine_layer(self, x, x_hat, wa, wb, b):
        return tf.math.tanh(wa * x + wb * x_hat + b)

    def call(self, x, x_hat):
        # forward propogation of the neural network
        h1 = self.mine_layer(x, x_hat, self.w1a, self.w1b, self.b1)
        h2 = self.mine_layer(x, x_hat, self.w2a, self.w2b, self.b2)
        out = self.w3 * (h1 + h2) + self.b3

        return out, tf.exp(out)


class CharacteristicFunction:
    '''
    class to construct a function that represents the characteristic function
    
    Parameters
    ----------
    size: the total sample size
    z_dims: the dimension of the conditional component
    x_dims: the dimension of the data input
    test_size: the size of testing samples
    '''

    def __init__(self, size, x_dims, z_dims, test_size):
        self.n_samples = size
        self.hidden_dims = 10
        self.test_size = test_size
        
        # define the structure of the neural network
        self.input_dim = z_dims + x_dims
        self.z_dims = z_dims
        self.x_dims = x_dims
        self.input_shape1x = [self.x_dims, self.hidden_dims]
        self.input_shape1z = [self.z_dims, self.hidden_dims]
        self.input_shape1 = [self.input_dim, self.hidden_dims]
        self.input_shape2 = [self.hidden_dims, 1]
        
        # initialize the weights of the neural network
        self.w1x = self.xavier_var_creator(self.input_shape1x)
        '''
        self.w1x_samples = tf.reshape(tf.tile(self.w1x, [self.test_size, self.n_samples]),
                                      [self.test_size, self.n_samples, self.hidden_dims])

        self.w1x_data = tf.reshape(tf.tile(self.w1x, [self.test_size, 1]),
                                   [self.test_size, self.x_dims, self.hidden_dims])
        self.w1z_ = self.xavier_var_creator(self.input_shape1z)
        self.w1z = tf.reshape(tf.tile(self.w1z_, [self.test_size, 1]),
                                      [self.test_size, self.z_dims, self.hidden_dims])
        self.w1 = tf.concat([self.w1x, self.w1z], axis=1)
        '''
        self.b1 = tf.squeeze(self.xavier_var_creator([self.x_dims, 1]))
        # self.b1 = tf.Variable(tf.zeros(self.input_shape1[1], tf.float64))

        self.w2 = self.xavier_var_creator(self.input_shape2)
        '''
        self.w2_samples = tf.reshape(tf.tile(self.w2, [self.test_size, 1]),
                                     [self.test_size, self.hidden_dims, 1])
        '''
        self.b2 = tf.Variable(tf.zeros(self.input_shape2[1], tf.float64))

    def xavier_var_creator(self, input_shape):
        # function used for initialization
        xavier_stddev = tf.sqrt(2.0 / (input_shape[0]))
        init = tf.random.normal(shape=input_shape, mean=0.0, stddev=xavier_stddev)
        init = tf.cast(init, tf.float64)
        var = tf.Variable(init, shape=tf.TensorShape(input_shape), trainable=True)
        return var

    def call(self, x, z):
        # forward propogation of the neural network
        # inputs are concatenations of z and v
        # z = tf.reshape(tensor=inputs, shape=[-1, self.input_dim])
        x = tf.reshape(tensor=x, shape=[self.test_size, -1, self.x_dims])
        z = tf.reshape(tensor=z, shape=[self.test_size, -1, self.z_dims])

        h1 = tf.nn.sigmoid(tf.matmul(x, self.w1x) + self.b1)
        # h1 = tf.nn.sigmoid(tf.matmul(z, self.w1z) + tf.matmul(x, self.w1x) + self.b1)
        out = tf.nn.sigmoid(tf.matmul(h1, self.w2))

        return out

