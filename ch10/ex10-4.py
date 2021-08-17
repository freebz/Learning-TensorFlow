# Loading weights

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np

DATA_DIR = '/tmp/data'


data = input_data.read_data_sets(DATA_DIR, one_hot=True)


class simple_cnn:
    def __init__(self, x_image, keep_prob, weights=None, sess=None):

        self.parameters = []
        self.x_image = x_image

        conv1 = self.conv_layer(x_image, shape=[5, 5, 1, 32])
        conv1_pool = self.max_pool_2x2(conv1)

        conv2 = self.conv_layer(conv1_pool, shape=[5, 5, 32, 64])
        conv2_pool = self.max_pool_2x2(conv2)

        conv2_flat = tf.reshape(conv2_pool, [-1, 7*7*64])
        full_1 = tf.nn.relu(self.full_layer(conv2_flat, 1024))

        full1_drop = tf.nn.dropout(full_1, keep_prob=keep_prob)

        self.y_conv = self.full_layer(full1_drop, 10)

        if weights is not None and sess is not None:
            self.load_weights(weights, sess)

            
    def weight_variable(self, shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial, name='weights')

    
    def bias_variable(self, shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial, name='biases')

    
    def conv2d(self, x, W):
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

    
    def max_pool_2x2(self, x):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                              strides=[1, 2, 2, 1], padding='SAME')

    
    def conv_layer(self, input, shape):
        W = self.weight_variable(shape)
        b = self.bias_variable([shape[3]])
        self.parameters += [W, b]
        return tf.nn.relu(self.conv2d(input, W) + b)

    
    def full_layer(self, input, size):
        in_size = int(input.get_shape()[1])
        W = self.weight_variable([in_size, size])
        b = self.bias_variable([size])
        self.parameters += [W, b]
        return tf.matmul(input, W) + b


    def load_weights(self, weights, sess):
        for i, w in enumerate(weights):
            print("Weight index: {}".format(i),
                  "Weight shape: {}".format(w.shape))
            sess.run(self.parameters[i].assign(w))





path = 'tmp//'




x = tf.placeholder(tf.float32, shape=[None, 784])
x_image = tf.reshape(x, [-1, 28, 28, 1])
y_ = tf.placeholder(tf.float32, shape=[None, 10])
keep_prob = tf.placeholder(tf.float32)

sess = tf.Session()

weights = np.load(path + 'cnn_weight_storage.npz', allow_pickle=True)
weights = list(weights.items())[0][1]
cnn = simple_cnn(x_image, keep_prob, weights, sess)

cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(
        logits=cnn.y_conv,
        labels=y_))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(cnn.y_conv, 1),
                              tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

X = data.test.images.reshape(10, 1000, 784)
Y = data.test.labels.reshape(10, 1000, 10)
test_accuracy = np.mean([sess.run(accuracy,
                                  feed_dict={x:X[i], y_:Y[i], keep_prob:1.0})
                         for i in range(10)])

sess.close()

print("test accuracy: {}".format(test_accuracy))



# Weight index: 0 Weight shape: (5, 5, 1, 32)
# Weight index: 1 Weight shape: (32,)
# Weight index: 2 Weight shape: (5, 5, 32, 64)
# Weight index: 3 Weight shape: (64,)
# Weight index: 4 Weight shape: (3136, 1024)
# Weight index: 5 Weight shape: (1024,)
# Weight index: 6 Weight shape: (1024, 10)
# Weight index: 7 Weight shape: (10,)
# test accuracy: 0.9880999326705933
