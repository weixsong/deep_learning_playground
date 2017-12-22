'''
Mixture Density Network is showed in this tutorial.
MDN (Mixture Density Network) is a deep nework but the output layer
will learn a distribution by GMM.
Author: Wei Song
'''


import numpy as np
import tensorflow as tf
import math
import matplotlib.pyplot as plt


# Our current model only predicts one output value for each input,
# so this approach will fail miserably. What we want is a model that has the capacity
# to predict a range of different output values for each input.
# In the next section we implement a Mixture Density Network (MDN) to do achieve this task.


# Begin the Mixture logistic distribution

def log_sum_exp(x):
    """
    numerically stable log_sum_exp implementation that prevents overflow
    log_sum_exp: log(sum(exp(x), axis=-1))
    """
    axis = len(x.get_shape()) - 1
    m = tf.reduce_max(x, axis)
    m2 = tf.reduce_max(x, axis, keep_dims=True)
    return m + tf.log(tf.reduce_sum(tf.exp(x - m2), axis))


def log_prob_from_logits(x):
    """ numerically stable log_softmax implementation that prevents overflow """
    axis = len(x.get_shape()) - 1
    m = tf.reduce_max(x, axis, keep_dims=True)
    return x - m - tf.log(tf.reduce_sum(tf.exp(x - m), axis, keep_dims=True))


tf.reset_default_graph()

NHIDDEN = 50
STDEV = 0.5
KMIX = 10  # number of mixtures
NOUT = KMIX * 3  # pi, mu, stdev

x = tf.placeholder(dtype=tf.float32, shape=[None,1], name="x")
y = tf.placeholder(dtype=tf.float32, shape=[None,1], name="y")

Wh = tf.Variable(tf.random_normal([1, NHIDDEN], stddev=STDEV, dtype=tf.float32))
bh = tf.Variable(tf.random_normal([1, NHIDDEN], stddev=STDEV, dtype=tf.float32))

Wo = tf.Variable(tf.random_normal([NHIDDEN, NOUT], stddev=STDEV, dtype=tf.float32))
bo = tf.Variable(tf.random_normal([1, NOUT], stddev=STDEV, dtype=tf.float32))

hidden_layer = tf.nn.tanh(tf.matmul(x, Wh) + bh)
output = tf.matmul(hidden_layer, Wo) + bo


def mix_logistic_loss(x, probs, means, scales):
    logit_probs = tf.log(probs)
    log_scales = tf.log(scales)
    log_scales = tf.maximum(log_scales, -7.)

    centered_x = x - means
    inv_stdv = tf.reciprocal(scales)

    mid_in = inv_stdv * centered_x
    log_pdf_mid = mid_in - log_scales - 2. * tf.nn.softplus(mid_in)

    log_probs = log_pdf_mid + log_prob_from_logits(logit_probs)
    final_log = log_sum_exp(log_probs)
    return -tf.reduce_mean(final_log)


def get_mixture_coef(output):
    out_pi, out_mu, out_sigma = tf.split(output, 3, 1)
    max_pi = tf.reduce_max(out_pi, 1, keep_dims=True)
    out_pi = tf.subtract(out_pi, max_pi)
    out_pi = tf.exp(out_pi)
    normalize_pi = tf.reciprocal(tf.reduce_sum(out_pi, 1, keep_dims=True))
    out_pi = tf.multiply(normalize_pi, out_pi)
    out_sigma = tf.exp(out_sigma)

    return out_pi, out_mu, out_sigma


out_pi, out_mu, out_sigma = get_mixture_coef(output)


NSAMPLE = 2500
y_data = np.float32(np.random.uniform(-10.5, 10.5, (1, NSAMPLE))).T
r_data = np.float32(np.random.normal(size=(NSAMPLE, 1)))  # random noise
x_data = np.float32(np.sin(0.75 * y_data) * 7.0 + y_data * 0.5 + r_data * 1.0)

plt.figure(figsize=(8, 8))
plt.plot(x_data, y_data, 'ro', alpha=0.3)
plt.show()

loss = mix_logistic_loss(y, out_pi, out_mu, out_sigma)
optimizer = tf.train.AdamOptimizer().minimize(loss)

# generate test data
x_test = np.float32(np.arange(-15, 15, 0.1))
NTEST = x_test.size
x_test = x_test.reshape(NTEST, 1)


def get_pi_idx(x, pdf):
    N = pdf.size
    accumulate = 0
    for i in range(0, N):
        accumulate += pdf[i]
        if accumulate >= x:
            return i
    print('error with sampling ensemble')
    return -1


# generate samples from the network learned distribution
def generate_ensemble(probs, means, scales, M=10):
    NTEST = x_test.size
    result = np.random.rand(NTEST, M)  # initially random [0, 1]
    rn = np.random.logistic(size=(NTEST, M))  # logistic random matrix
    mu = 0
    std = 0
    idx = 0

    # transforms result into random ensembles
    for j in range(0, M):
        for i in range(0, NTEST):
            idx = get_pi_idx(result[i, j], probs[i])
            mu = means[i, idx]
            std = scales[i, idx]
            result[i, j] = mu + rn[i, j] * std

    return result


# training
NEPOCH = 20000

# Initializing the variables
init = tf.global_variables_initializer()

losses = []  # store the training progress here.
# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    for i in range(NEPOCH):
        l, _ = sess.run([loss, optimizer], feed_dict={x: x_data, y: y_data})
        losses.append(l)
        if i % 20 == 0:
            print("MDN loss: %s, step is %s" % (str(l), str(i)))

    plt.figure(figsize=(8, 8))
    plt.plot(np.arange(len(losses)), losses, 'r-')
    plt.title("loss")
    plt.show()

    # generate samples
    test_probs, test_means, test_scales = sess.run(get_mixture_coef(output), feed_dict={x: x_test})
    y_test = generate_ensemble(test_probs, test_means, test_scales)
    plt.figure(figsize=(8, 8))
    plt.plot(x_data, y_data, 'ro', x_test, y_test, 'bo', alpha=0.3)
    plt.show()

    print("job done")

