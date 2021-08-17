import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import os

DATA_DIR = 'tmp/data'
data = input_data.read_data_sets(DATA_DIR, one_hot=True)

DIR = 'saved_model/'

tf.reset_default_graph()
x = tf.placeholder(tf.float32, [None, 784], name='x')
W = tf.Variable(tf.zeros([784, 10]), name='W')
y_true = tf.placeholder(tf.float32, [None, 10])
y_pred = tf.matmul(x, W)
cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(logits=y_pred,
                                            labels=y_true))
gd_step = tf.train.GradientDescentOptimizer(0.5)\
    .minimize(cross_entropy)
correct_mask = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y_true, 1))
accuracy = tf.reduce_mean(tf.cast(correct_mask, tf.float32))

saver = tf.train.Saver()

with tf.Session() as sess:

    saver.restore(sess, os.path.join(DIR, "model_ckpt-1000"))
    ans = sess.run(accuracy, feed_dict={x: data.test.images,
                                        y_true: data.test.labels})

print("Accuracy: {:.4}%".format(ans*100))



# Accuracy: 92.02%
