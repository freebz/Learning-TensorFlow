import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import os

DATA_DIR = 'tmp/data'
data = input_data.read_data_sets(DATA_DIR, one_hot=True)



tf.reset_default_graph()
DIR = "saved_model/"

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    saver = tf.train.import_meta_graph(os.path.join(
        DIR, "model_ckpt-1000.meta"))
    saver.restore(sess, os.path.join(DIR, "model_ckpt-1000"))
    x = tf.get_collection('train_var')[0]
    y_true = tf.get_collection('train_var')[1]
    accuracy = tf.get_collection('train_var')[2]
    
    ans = sess.run(accuracy, feed_dict={x: data.test.images,
                                        y_true: data.test.labels})

print("Accuracy: {:.4}%".format(ans*100))



# Accuracy: 91.52%
