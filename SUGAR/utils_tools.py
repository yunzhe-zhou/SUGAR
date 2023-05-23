import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import os
import time
import math
from datetime import datetime
import logging
from sklearn.metrics.pairwise import rbf_kernel
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp
from scipy.stats import wilcoxon
import random
from scipy import stats
from collections import defaultdict
import warnings
import SUGAR.cit_gan as cit_gan
from scipy.stats import rankdata
import xlwt
from tempfile import TemporaryFile
import scipy
import SUGAR.gan_utils as gan_utils
tf.random.set_seed(42)
np.random.seed(42)


# =============================================================================
# This provides the helper function for the GAN with Sinkhorn divergence
# =============================================================================

'''
Inputs:
 - z: Confounder variables, this is the conditioning set
 - x: First target variable
 - y: Second target variable
'''


def gcit_tools(x_train,z_train,x_test,z_test,v_dims,h_dims, M = 200, batch_size=64, n_iter=1000, standardise = True,normalize=True):
    """
    This provides the helper function for the GAN with Sinkhorn divergence
    
    Input
    ----------
    x_train: the training data of X
    z_train: the training data of the conditional Z
    x_test: the testing data of X
    z_test: the testing data of the conditional Z
    v_dims: the dimension of the random noise
    h_dims: the dimension of the neural network
    M: the number of pseudo samples
    batch_size: the batch size for training
    n_iter: the total number of iterations for training
    standardise: whether standardizing the data or not
    normalize: whether normalizing the data or not
    
    Output
    ----------
    result: contains the generated samples by GAN and it corresponded test data for each fold
    """
    
    if normalize:
        # normalize the data
        X=np.concatenate((x_train,x_test),axis=0)
        Z=np.concatenate((z_train,z_test),axis=0)
        x_train=(x_train-X.min())/(X.max()-X.min())
        x_test=(x_test-X.min())/(X.max()-X.min())
        z_train=(z_train-Z.min())/(Z.max()-Z.min())
        z_test=(z_test-Z.min())/(Z.max()-Z.min())       
    x_dims = x_train.shape[1]
    z_dim = z_train.shape[1]
    n=int(x_train.shape[0]*2)
    # build data pipline for training set
    dataset1 = tf.data.Dataset.from_tensor_slices((x_train, z_train))
    testset1 = tf.data.Dataset.from_tensor_slices((x_test, z_test))
    batched_test1 = dataset1.batch(1)
    batched_test2 = testset1.batch(1)
    # Repeat n epochs
    epochs = int(n_iter)
    dataset1 = dataset1.repeat(epochs)
    batched_train1 = dataset1.shuffle(300).batch(batch_size)
    batched_training_set1 = dataset1.shuffle(300).batch(batch_size)

    data_k = [[batched_train1, batched_test1, batched_test2]]

    # no. of random and hidden dimensions
#     v_dims = int(5)
#     h_dims = int(1000)
    v_dims = v_dims
    h_dims = h_dims
    
    # specify the distribution for the input noise
    v_dist = tfp.distributions.Normal(0, scale=tf.sqrt(1.0 / 3.0))

    # create instance of G & D
    lr = 1e-4
    # input_dims = x_train.shape[1]
    
    # specify the generator and discriminator for the sinkhorn
    generator_x = cit_gan.WGanGenerator(n, z_dim, h_dims, v_dims, batch_size)
    discriminator_x = cit_gan.WGanDiscriminator(n, z_dim, h_dims, x_dims, batch_size)
    
    # specify the choices of hyperparameters
    gen_clipping_val = 0.5
    gen_clipping_norm = 1.0
    w_clipping_val = 0.5
    w_clipping_norm = 1.0
    scaling_coef = 1000.0
    sinkhorn_eps = 0.8
    sinkhorn_l = 30
    
    # specify the optimizer model
    gx_optimiser = tf.keras.optimizers.Adam(lr, beta_1=0.5, clipnorm=gen_clipping_norm, clipvalue=gen_clipping_val)
    dx_optimiser = tf.keras.optimizers.Adam(lr, beta_1=0.5, clipnorm=w_clipping_norm, clipvalue=w_clipping_val)
    
    # helper function for sinkhorn updates of the discriminative network
    
    @tf.function(experimental_relax_shapes=True)
    def x_update_d(real_x, real_x_p, real_z, real_z_p, v, v_p):
        gen_inputs = tf.concat([real_z, v], axis=1)
        gen_inputs_p = tf.concat([real_z_p, v_p], axis=1)
        # concatenate real inputs for WGAN discriminator (x, z)
        d_real = tf.concat([real_x, real_z], axis=1)
        d_real_p = tf.concat([real_x_p, real_z_p], axis=1)
        fake_x = generator_x.call(gen_inputs)
        fake_x_p = generator_x.call(gen_inputs_p)
        d_fake = tf.concat([fake_x, real_z], axis=1)
        d_fake_p = tf.concat([fake_x_p, real_z_p], axis=1)

        with tf.GradientTape() as disc_tape:
            f_real = discriminator_x.call(d_real)
            f_fake = discriminator_x.call(d_fake)
            f_real_p = discriminator_x.call(d_real_p)
            f_fake_p = discriminator_x.call(d_fake_p)
            # call compute loss using @tf.function + autograph

            loss1 = \
                gan_utils.benchmark_loss(f_real, f_fake, scaling_coef, sinkhorn_eps, sinkhorn_l, f_real_p, f_fake_p)[0]
            # disc_loss = - tf.math.minimum(loss1, 1)
            disc_loss = - loss1
        # update discriminator parameters
        d_grads = disc_tape.gradient(disc_loss, discriminator_x.trainable_variables)
        dx_optimiser.apply_gradients(zip(d_grads, discriminator_x.trainable_variables))
    
    # helper function for sinkhorn updates of the generative network
    
    @tf.function(experimental_relax_shapes=True)
    def x_update_g(real_x, real_x_p, real_z, real_z_p, v, v_p):
        gen_inputs = tf.concat([real_z, v], axis=1)
        gen_inputs_p = tf.concat([real_z_p, v_p], axis=1)
        # concatenate real inputs for WGAN discriminator (x, z)
        d_real = tf.concat([real_x, real_z], axis=1)
        d_real_p = tf.concat([real_x_p, real_z_p], axis=1)
        with tf.GradientTape() as gen_tape:
            fake_x = generator_x.call(gen_inputs)
            fake_x_p = generator_x.call(gen_inputs_p)
            d_fake = tf.concat([fake_x, real_z], axis=1)
            d_fake_p = tf.concat([fake_x_p, real_z_p], axis=1)
            f_real = discriminator_x.call(d_real)
            f_fake = discriminator_x.call(d_fake)
            f_real_p = discriminator_x.call(d_real_p)
            f_fake_p = discriminator_x.call(d_fake_p)
            # call compute loss using @tf.function + autograph
            gen_loss, loss_xy, loss_xx, loss_yy = gan_utils.benchmark_loss(f_real, f_fake, scaling_coef, sinkhorn_eps,
                                                                           sinkhorn_l, f_real_p, f_fake_p)
        # update generator parameters
        generator_grads = gen_tape.gradient(gen_loss, generator_x.trainable_variables)
        gx_optimiser.apply_gradients(zip(generator_grads, generator_x.trainable_variables))
        return gen_loss, loss_xy, loss_xx, loss_yy


    x_samples_all1 = []
    x_samples_all2 = []
    x_all1 = []
    x_all2 = []
    test_size = z_test.shape[0]
    
    # for each batched sample
    for batched_trainingset, batched_testset1, batched_testset2 in data_k:
        for x_batch1, z_batch1 in batched_trainingset.take(n_iter):
            for x_batch2, z_batch2 in batched_training_set1.take(1):
                if x_batch1.shape[0] != batch_size:
                    continue
                # sample noise v
                noise_v = v_dist.sample([batch_size, v_dims])
                noise_v = tf.cast(noise_v, tf.float64)
                noise_v_p = v_dist.sample([batch_size, v_dims])
                noise_v_p = tf.cast(noise_v_p, tf.float64)
                # update the discriminator 
                x_update_d(x_batch1, x_batch2, z_batch1, z_batch2, noise_v, noise_v_p)
                # update the generator
                loss_x, a, b, c = x_update_g(x_batch1, x_batch2, z_batch1, z_batch2, noise_v, noise_v_p)

        x_samples1 = []
        x1 = []
        x_samples2 = []
        x2 = []

        # the following code generate pseudo data for all B and it takes 61 secs for one test
        for test_x, test_z in batched_testset1:
            tiled_z = tf.tile(test_z, [M, 1])
            noise_v = v_dist.sample([M, v_dims])
            noise_v = tf.cast(noise_v, tf.float64)
            g_inputs = tf.concat([tiled_z, noise_v], axis=1)
            # generator samples from G and evaluate from D
            fake_x = generator_x.call(g_inputs, training=False)
            x_samples1.append(fake_x)
            x1.append(test_x)
        
        # normalize the generated data back to the scaling of the original data
        if normalize:
            x_samples1=x_samples1*(tf.math.reduce_max(X)-tf.math.reduce_min(X))+tf.math.reduce_min(X)
            x1=x1*(tf.math.reduce_max(X)-tf.math.reduce_min(X))+tf.math.reduce_min(X)
            
        x_samples_all1.append(x_samples1)
        x_all1.append(x1)
        
        # the following code generate pseudo data for all B and it takes 61 secs for one test
        for test_x, test_z in batched_testset2:
            tiled_z = tf.tile(test_z, [M, 1])
            noise_v = v_dist.sample([M, v_dims])
            noise_v = tf.cast(noise_v, tf.float64)
            g_inputs = tf.concat([tiled_z, noise_v], axis=1)
            # generator samples from G and evaluate from D
            fake_x = generator_x.call(g_inputs, training=False)
            x_samples2.append(fake_x)
            x2.append(test_x)
        
        # normalize the generated data back to the scaling of the original data
        if normalize:
            x_samples2=x_samples2*(tf.math.reduce_max(X)-tf.math.reduce_min(X))+tf.math.reduce_min(X)
            x2=x2*(tf.math.reduce_max(X)-tf.math.reduce_min(X))+tf.math.reduce_min(X)
              
        x_samples_all2.append(x_samples2)
        x_all2.append(x2)

    result=[]
    result.append(x_samples_all1)
    result.append(x_samples_all2)
    result.append(x_all1)
    result.append(x_all2)
    return result




