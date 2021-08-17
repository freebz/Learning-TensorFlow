# 사용자 정의 손실 함수

import numpy as np
import tensorflow as tf

LAMBDA = 1e-5

def mul_lambda(val):
    return np.multiply(val, LAMBDA).astype(np.float32)

@tf.RegisterGradient("PyMulLambda")
def grad_mul_lambda(op, grad):
    return LAMBDA*grad


x = tf.placeholder(tf.float32, [None, 784])
W = tf.Variable(tf.zeros([784, 10]))

y_true = tf.placeholder(tf.float32, [None, 10])
y_pred = tf.matmul(x, W)

cross_entropy = \
    tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits\
                   (logits=y_pred, labels=y_true))

with tf.get_default_graph().gradient_override_map({"PyFunc": "PyMulLambda"}):
    total_loss = cross_entropy + \
        tf.py_func(mul_lambda, [tf.nn.l2_loss(W)], [tf.float32])[0]

gd_step = tf.train.GradientDescentOptimizer(0.5).minimize(total_loss)

correct_mask = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y_true, 1))
accuracy = tf.reduce_mean(tf.cast(correct_mask, tf.float32))
