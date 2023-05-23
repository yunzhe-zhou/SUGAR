# Utilites related to Sinkhorn computations and training for TensorFlow 2.0

import numpy as np
import tensorflow as tf


def condition(i, x, y, scaling_coef, tensor):
    '''
    :param i: current index
    :return: boolean
    '''
    # number of data points for iteration
    m = y.shape[0]
    return tf.less(i, m)


def write_to_row(j, x, y, scaling_coef, tensor):
    '''
    :param j: current index of y
    :param x: a tensor of shape [i, seq_len]
    :param y: a tensor of shape [m, seq_len]
    :param scaling_coef: a scaling coefficient
    :param tensor: tensor to write
    :return: while loop body
    '''
    tensor = tensor.write(j, scaling_coef * tf.reduce_sum(tf.math.squared_difference(x, y[j])))
    return [tf.add(j, 1), x, y, scaling_coef, tensor]


def cost_xy(x, y, scaling_coef):
    '''
    :param x: x is tensor of shape [batch_size, sequence length]
    :param y: y is tensor of shape [batch_size, sequence length]
    :param scaling_coef: a scaling coefficient for distance between x and y
    :return: cost matrix: a matrix of size [batch_size, batch_size] where
      c_{ij} = c(x^i, y^j) = \sum_t^T |x^i_t - y^j_t|
    '''
    m = x.shape[0]

    C_xy = tf.TensorArray(tf.float32, size=1, dynamic_size=True, clear_after_read=False)

    for i in tf.range(m):
        j = tf.constant(0)
        row_tensor = tf.TensorArray(tf.float32, size=m, clear_after_read=False)
        # write to row
        _, _, _, _, row_vals = tf.while_loop(condition, write_to_row, [j, x[i], y, scaling_coef, row_tensor])
        row_vals = row_vals.stack()
        # write to cost matrix
        C_xy = C_xy.write(i, row_vals)
    C_xy = C_xy.stack()

    return C_xy

def cost_xy(x, y, scaling_coef):
    '''
    L2 distance between vectors, using expanding and hence is more memory intensive
    :param x: x is tensor of shape [batch_size, sequence length]
    :param y: y is tensor of shape [batch_size, sequence length]
    :param scaling_coef: a scaling coefficient for distance between x and y
    :return: cost matrix: a matrix of size [batch_size, batch_size] where
    '''
    x = tf.expand_dims(x, 1)
    y = tf.expand_dims(y, 0)
    return tf.reduce_sum((x - y)**2, -1) * scaling_coef

@tf.function(experimental_relax_shapes=True)
def modified_cost(x, y, h, M, scaling_coef):
    '''
    :param x: a tensor of shape [batch_size, sequence length]
    :param y: a tensor of shape [batch_size, sequence length]
    :param h: a tensor of shape [batch_size, sequence length]
    :param M: a tensor of shape [batch_size, sequence length]
    :param scaling_coef: a scaling coefficient for squared distance between x and y
    :return: L1 cost matrix plus h, M modification:
    a matrix of size [batch_size, batch_size] where
    c_hM_{ij} = c_hM(x^i, y^j) = L1_cost + \sum_{t=1}^{T-1}h_t\Delta_{t+1}M ====> NOTE: T-1 here, T = # of time steps
    '''
    
    # compute sum_{t=1}^{T-1} h[t]*(M[t+1]-M[t])
    DeltaMt = M[:,1:] - M[:,:-1]
    ht      = h[:, :-1]
    C_hM    = tf.reduce_sum(ht[:,None,:] * DeltaMt[None,:,:], -1) * scaling_coef

    # Compute L1 cost $\sum_t^T |x^i_t - y^j_t|$
    l1_cost_matrix = cost_xy(x, y, scaling_coef)
    # compute the mean of cost matrix c_xy and c_hm and add to tensorboard for monitoring training process
    l1 = tf.reduce_mean(l1_cost_matrix)
    c = tf.reduce_mean(C_hM)
    abs_c = tf.reduce_mean(tf.abs(C_hM))

    return tf.math.add(l1_cost_matrix, C_hM), l1, c, abs_c

@tf.function
def benchmark_sinkhorn(x, y, scaling_coef, epsilon=1.0, L=10):
    '''
    :param x: a tensor of shape [batch_size, sequence length]
    :param y: a tensor of shape [batch_size, sequence length]
    :param scaling_coef: a scaling coefficient for squared distance between x and y
    :param epsilon: (float) entropic regularity constant
    :param L: (int) number of iterations
    :return: V: (float) value of regularized optimal transport
    '''
    n_data = x.shape[0]
    # Note that batch size of x can be different from batch size of y
    m = 1.0 / tf.cast(n_data, tf.float32) * tf.ones(n_data, dtype=tf.float32)
    n = 1.0 / tf.cast(n_data, tf.float32) * tf.ones(n_data, dtype=tf.float32)
    m = tf.expand_dims(m, axis=1)
    n = tf.expand_dims(n, axis=1)

    c_xy = cost_xy(x, y, scaling_coef)  # shape: [batch_size, batch_size]

    k = tf.exp(-c_xy / epsilon) + 1e-09 # add 1e-09 to prevent numerical issues
    k_t = tf.transpose(k)

    a = tf.expand_dims(tf.ones(n_data, dtype=tf.float32), axis=1)
    b = tf.expand_dims(tf.ones(n_data, dtype=tf.float32), axis=1)
    for i in tf.range(L):
        b = m / tf.matmul(k_t, a)  # shape: [m,]
        a = n / tf.matmul(k, b)  # shape: [m,]

    return tf.reduce_sum(a * k * tf.reshape(b, (1, -1)) * c_xy)


@tf.function
def benchmark_sinkhorn(x, y, scaling_coef, epsilon=1.0, L=10, Lmin=5):

    """
    Given two emprical measures with n points each with locations x and y
    outputs an approximation of the OT cost with regularization parameter epsilon
    niter is the max. number of steps in sinkhorn loop
    """

    n_data = x.shape[0]

    # The Sinkhorn algorithm takes as input three variables :
    C = cost_xy(x, y, scaling_coef)  # Wasserstein cost function

    # both marginals are fixed with equal weights
    mu = 1.0 / tf.cast(n_data, tf.float64) * tf.ones(n_data, dtype=tf.float64)
    nu = 1.0 / tf.cast(n_data, tf.float64) * tf.ones(n_data, dtype=tf.float64)

    # Parameters of the Sinkhorn algorithm.
    thresh = 10**(-2) # stopping criterion
    thresh = tf.cast(thresh, tf.float64)

    # Elementary operations .....................................................................

    def M(u, v):
        "Modified cost for logarithmic updates"
        "$M_{ij} = (-c_{ij} + u_i + v_j) / \epsilon$"
        return (-C + u[:,None] + v[None,:]) / epsilon

    def lse(A):
        "log-sum-exp"
        return tf.math.reduce_logsumexp(A, axis=1, keepdims=True)
        #return tf.math.log(tf.reduce_sum(tf.exp(A), axis=1, keepdims=True) + 1e-6)  # add 10^-6 to prevent NaN

    # Actual Sinkhorn loop ......................................................................
    u, v, err = 0. * mu, 0. * nu, 0.
    err = tf.cast(err, tf.float64)
    actual_nits = 0  # to check if algorithm terminates because of threshold or max iterations reached

    for i in tf.range(L):
        u1 = u  # useful to check the update
        u = epsilon * (tf.math.log(mu) - tf.squeeze(lse(M(u, v)))) + u
        v = epsilon * (tf.math.log(nu) - tf.squeeze(lse(tf.transpose(M(u, v))))) + v
        err = tf.reduce_sum(tf.math.abs(u - u1))

        actual_nits += 1
        if tf.math.greater(thresh, err) and i >= Lmin:
            break
    U, V = u, v
    pi = tf.exp(M(U, V))  # Transport plan pi = diag(a)*K*diag(b)
    cost = tf.reduce_sum(pi * C)  # Sinkhorn cost

    return cost, actual_nits

@tf.function(experimental_relax_shapes=True)
def compute_sinkhorn(x, y, M, h, scaling_coef, epsilon=1.0, L=10, Lmin = 5):

    """
    Given two emprical measures with n points each with locations x and y
    outputs an approximation of the OT cost with regularization parameter epsilon
    niter is the max. number of steps in sinkhorn loop
    """

    n_data = x.shape[0]

    # The Sinkhorn algorithm takes as input three variables :
    C, l1, c, abs_c = modified_cost(x, y, h, M, scaling_coef)  # shape: [batch_size, batch_size]

    # both marginals are fixed with equal weights, have to append dimension otherwise weird tf bugs
    mu = 1.0 / n_data * tf.ones(n_data, dtype=tf.float32)
    nu = 1.0 / n_data * tf.ones(n_data, dtype=tf.float32)
    mu = tf.expand_dims(mu, 1)
    nu = tf.expand_dims(nu, 1)

    # Parameters of the Sinkhorn algorithm.
    thresh = 10**(-2)  # stopping criterion

    # Actual Sinkhorn loop ......................................................................
    u, v, err = 0. * mu, 0. * nu, 0.
    actual_nits = 0  # to check if algorithm terminates because of threshold or max iterations reached
    
    for i in tf.range(L):
        u1 = u  # useful to check the update
        Muv = (-C + u + tf.transpose(v)) / epsilon
        u = epsilon * (tf.math.log(mu) - (tf.reduce_logsumexp(Muv, axis=1, keepdims=True))) + u
        Muv = (-C + u + tf.transpose(v)) / epsilon
        v = epsilon * (tf.math.log(nu) - (tf.reduce_logsumexp(tf.transpose(Muv), axis=1, keepdims=True))) + v
        err = tf.reduce_sum(tf.math.abs(u - u1))

        actual_nits += 1
        if tf.math.greater(thresh, err) and actual_nits >= Lmin:
            break
    U, V = u, v
    Muv = (-C + u + tf.transpose(v)) / epsilon
    pi = tf.exp(Muv)  # Transport plan pi = diag(a)*K*diag(b)
    cost = tf.reduce_sum(pi * C)  # Sinkhorn cost

    return cost, l1, c, abs_c, actual_nits

def compute_N(M):
    '''
    :param M: A tensor of shape (batch_size, sequence length)
    :return: A tensor of shape (m, sequence length - 1)
    '''
    T = M.shape[1]
    M_shift = M[:, 1:]
    M_trunc = M[:, :T - 1]
    return tf.math.subtract(M_shift, M_trunc)


def martingale_regularization(M, reg_lam, reg_gamma, scaling_coef):
    '''
    Compute the regularization for the martingale condition (i.e. p_M).
    :param M: a tensor of shape (batch_size, sequence length), the output of an RNN applied to X
    :param reg_lam: scale parameter for first term in pM
    :param reg_gamma: scale parameter for second term in pM
    :param scaling_coef: a scaling coefficient, should be the same as for squared distance between x and y
    :return: A rank 0 tensors (i.e. scalers)
    This tensor represents the martingale penalization term denoted $p_M$
    '''
    m = M.shape[0]
    m = tf.cast(m, tf.float32)
    # compute delta M matrix N
    N = compute_N(M)

    # Compute \sum_i^m(\delta M)
    sum_m = tf.reduce_sum(N, axis=0)
    # Compute martingale penalty: P_M1 =  \sum_i^T(|\sum_i^m(\delta M)|) * scaling_coef
    sum_across_paths = tf.reduce_sum(tf.math.abs(sum_m)) * scaling_coef

    # compute P_M2 = \sum_i^m \sum_t^T-1 |\delta M| * scaling_coef to
    # encourage the difference in a single path to be none-zero
    sum_single = tf.reduce_sum(tf.math.abs(N), axis=1)
    sum_single_path = tf.reduce_sum(sum_single) * scaling_coef

    # the total pM term
    # parameters before two pM terms are used to balance the two terms as pM2 is much larger than pM1
    pm = reg_lam * sum_across_paths - reg_gamma * sum_single_path

    # return the total pm term for computations and the other two terms for monitoring reasons
    return pm, sum_across_paths, sum_single_path


def new_martingale_regularization(M, reg_lam, reg_gamma, scaling_coef):
    '''
    Compute the regularization for the martingale condition (i.e. p_M).
    :param M: a tensor of shape (batch_size, sequence length), the output of an RNN applied to X
    :param reg_lam: scale parameter for first term in pM
    :param reg_gamma: scale parameter for second term in pM
    :param scaling_coef: a scaling coefficient, should be the same as for squared distance between x and y
    :return: A rank 0 tensors (i.e. scalers)
    This tensor represents the martingale penalization term denoted $p_M$
    '''
    m = M.shape[0]
    m = tf.cast(m, tf.float32)
    # compute delta M matrix N
    N = compute_N(M)

    mt_m1 = M - tf.expand_dims(M[..., 0], axis=1)

    pm = tf.reduce_sum(tf.math.abs(tf.reduce_sum(N / mt_m1[..., 1:], axis=0))) * scaling_coef * reg_lam

    # Compute \sum_i^m(\delta M)
    sum_m = tf.reduce_sum(N, axis=0)
    # Compute martingale penalty: P_M1 =  \sum_i^T(|\sum_i^m(\delta M)|) * scaling_coef
    sum_across_paths = tf.reduce_sum(tf.math.abs(sum_m)) * scaling_coef
    sum_single = tf.reduce_sum(tf.math.abs(N), axis=1)
    sum_single_path = tf.reduce_sum(sum_single) * scaling_coef

    # return the total pm term for computations and the other two terms for monitoring reasons
    return pm, sum_across_paths, sum_single_path

def scale_invariante_martingale_regularization(M, reg_lam, reg_gamma, scaling_coef):
    '''
    Compute the regularization for the martingale condition (i.e. p_M).
    :param M: a tensor of shape (batch_size, sequence length), the output of an RNN applied to X
    :param reg_lam: scale parameter for first term in pM
    :param reg_gamma: scale parameter for second term in pM
    :param scaling_coef: a scaling coefficient, should be the same as for squared distance between x and y
    :return: A rank 0 tensors (i.e. scalers)
    This tensor represents the martingale penalization term denoted $p_M$
    '''
    m = M.shape[0]
    m = tf.cast(m, tf.float32)
    # compute delta M matrix N
    N = compute_N(M)
    N = N / tf.math.reduce_std(M)

    # Compute \sum_i^m(\delta M)
    sum_m = tf.reduce_sum(N, axis=0)
    # Compute martingale penalty: P_M1 =  \sum_i^T(|\sum_i^m(\delta M)|) * scaling_coef
    sum_across_paths = tf.reduce_sum(tf.math.abs(sum_m)) * scaling_coef / m

    # compute P_M2 = \sum_i^m \sum_t^T-1 |\delta M| * scaling_coef to
    # encourage the difference in a single path to be none-zero
    sum_single = tf.reduce_sum(tf.math.abs(N), axis=1)
    sum_single_path = tf.reduce_sum(sum_single) * scaling_coef

    # the total pM term
    # parameters before two pM terms are used to balance the two terms as pM2 is much larger than pM1
    pm = reg_lam * sum_across_paths 

    # return the total pm term for computations and the other two terms for monitoring reasons
    return pm, sum_across_paths, sum_single_path


def compute_loss(f_real, f_fake, m_real, m_fake, h_fake, scaling_coef, sinkhorn_eps, sinkhorn_l, f_real_p, f_fake_p, h_real_p, h_fake_p):
    '''
    :param x: real data of shape [batch size, sequence length]
    :param y: fake data of shape [batch size, sequence length]
    :param h: h(y) of shape [batch size, sequence length]
    :param m: M(x) of shape [batch size, sequence length]
    :param scaling_coef: a scaling coefficient
    :param sinkhorn_eps: Sinkhorn parameter - epsilon
    :param sinkhorn_l: Sinkhorn parameter - the number of iterations
    :return: final Sinkhorn loss(and several values for monitoring the training process)
    '''
    f_real = tf.reshape(f_real, [f_real.shape[0], -1])
    f_fake = tf.reshape(f_fake, [f_fake.shape[0], -1])
    f_real_p = tf.reshape(f_real_p, [f_real_p.shape[0], -1])
    f_fake_p = tf.reshape(f_fake_p, [f_fake_p.shape[0], -1])
    loss_xy, l1, chm, abs_chm, nit = compute_sinkhorn(f_real, f_fake, m_real, h_fake, scaling_coef, sinkhorn_eps, sinkhorn_l)
    loss_xx, _, _, _, _ = compute_sinkhorn(f_real, f_real_p, m_real, h_real_p, scaling_coef, sinkhorn_eps, sinkhorn_l)
    loss_yy, _, _, _, _ = compute_sinkhorn(f_fake, f_fake_p, m_fake, h_fake_p, scaling_coef, sinkhorn_eps, sinkhorn_l)

    loss = loss_xy - 0.5 * loss_xx - 0.5 * loss_yy
    # loss = loss_xy # - 0.5 * loss_xx - 0.5 * loss_yy

    return loss, l1, chm, abs_chm, nit


def benchmark_loss(x, y, scaling_coef, sinkhorn_eps, sinkhorn_l, xp=None, yp=None):
    '''
    :param x: real data of shape [batch size, sequence length]
    :param y: fake data of shape [batch size, sequence length]
    :param scaling_coef: a scaling coefficient
    :param sinkhorn_eps: Sinkhorn parameter - epsilon
    :param sinkhorn_l: Sinkhorn parameter - the number of iterations
    :return: final Sinkhorn loss(and several values for monitoring the training process)
    '''
    if yp is None:
        yp = y
    if xp is None:
        xp = x
    x = tf.reshape(x, [x.shape[0], -1])
    y = tf.reshape(y, [y.shape[0], -1])
    xp = tf.reshape(xp, [xp.shape[0], -1])
    yp = tf.reshape(yp, [yp.shape[0], -1])
    loss_xy, nitxy = benchmark_sinkhorn(x, y, scaling_coef, sinkhorn_eps, sinkhorn_l)
    loss_xx, nitxx = benchmark_sinkhorn(x, xp, scaling_coef, sinkhorn_eps, sinkhorn_l)
    loss_yy, nityy = benchmark_sinkhorn(y, yp, scaling_coef, sinkhorn_eps, sinkhorn_l)

    loss = loss_xy - 0.5 * loss_xx - 0.5 * loss_yy

    return loss, nitxy, nitxx, nityy
