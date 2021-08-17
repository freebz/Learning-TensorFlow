# Saver class = save and restore weights only

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import os

DATA_DIR = 'tmp/data'
data = input_data.read_data_sets(DATA_DIR, one_hot=True)

NUM_STEPS = 1000
MINIBATCH_SIZE = 100

DIR = 'saved_model/'

x = tf.placeholder(tf.float32, [None, 784], name='x')
W = tf.Variable(tf.zeros([784, 10]), name='W')
y_true = tf.placeholder(tf.float32, [None, 10])
y_pred = tf.matmul(x, W)
cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(logits=y_pred,
                                            labels=y_true))
gd_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
correct_mask = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y_true, 1))
accuracy = tf.reduce_mean(tf.cast(correct_mask, tf.float32))

saver = tf.train.Saver(max_to_keep=7,
                       keep_checkpoint_every_n_hours=1)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for step in range(1, NUM_STEPS +1):
        batch_xs, batch_ys = data.train.next_batch(MINIBATCH_SIZE)
        sess.run(gd_step, feed_dict={x: batch_xs, y_true: batch_ys})

        if step % 50 == 0:
            saver.save(sess, os.path.join(DIR, "model_ckpt"),
                       global_step=step)

    ans = sess.run(accuracy, feed_dict={x: data.test.images,
                                        y_true: data.test.labels})

print("Accuracy: {:.4}%".format(ans*100))



# Accuracy: 92.02%
