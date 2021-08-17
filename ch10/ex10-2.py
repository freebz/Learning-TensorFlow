# Load weights

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np

DATA_DIR = 'tmp/dta'
path = 'tmp//'

data = input_data.read_data_sets(DATA_DIR, one_hot=True)


loaded_w = np.load(path + 'weight_storage.npz')
loaded_w = list(loaded_w.items())[0][1]

x = tf.placeholder(tf.float32, [None, 784])
W = tf.Variable(tf.zeros([784, 10]))
y_true = tf.placeholder(tf.float32, [None, 10])
y_pred = tf.matmul(x, W)
cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(logits=y_pred, labels=y_true))
gd_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
correct_mask = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y_true, 1))
accuracy = tf.reduce_mean(tf.cast(correct_mask, tf.float32))

with tf.Session() as sess:
    # 로드한 가중치를 할당
    sess.run(W.assign(loaded_w))
    acc = sess.run(accuracy, feed_dict={x: data.test.images,
                                        y_true: data.test.labels})

print("Accuracy: {}".format(acc))



# Accuracy: 0.914900004863739
