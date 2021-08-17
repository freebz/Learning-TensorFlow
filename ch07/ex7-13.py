# TF-Slim

import sys
sys.path.append("./models/research/slim")



from datasets import dataset_utils
import tensorflow as tf

target_dir = './vgg_16'



import os
from urllib.request import urlopen

#url = ("http://localhost/car.jpg")
url = ("file://" + os.path.abspath("./images/car.jpg"))
im_as_string = urlopen(url).read()
image = tf.image.decode_jpeg(im_as_string, channels=3)



# filename_queue = tf.train.string_input_producer(
#     tf.train.match_filenames_once("./images/*.jpg"))
# image_reader = tf.WholeFileReader()
# _, image_file = image_reader.read(filename_queue)
# image = tf.image.decode_jpeg(image_file)



from nets import vgg
image_size = vgg.vgg_16.default_image_size


from preprocessing import vgg_preprocessing
processed_im = vgg_preprocessing.preprocess_image(image,
                                                  image_size,
                                                  image_size,
                                                  is_training=False)

processed_images = tf.expand_dims(processed_im, 0)



from tensorflow.contrib import slim

with slim.arg_scope(vgg.vgg_arg_scope()):
    logits, _ = vgg.vgg_16(processed_images,
                           num_classes=1000,
                           is_training=False)
probabilities = tf.nn.softmax(logits)

def vgg_arg_scope(weight_decay=0.0005):
    with slim.arg_scope([slim.conv2d, slim.fully_connected],
                        activation_fn=tf.nn.relu,
                        weights_regularizer=slim.l2_regularizer(weight_decay),
                        biases_initializer=tf.zeros_initializer):
        with slim.arg_scope([slim.conv2d], padding='SAME') as arg_sc:
            return arg_sc



import os

load_vars = slim.assign_from_checkpoint_fn(
    os.path.join(target_dir, 'vgg_16.ckpt'),
    slim.get_model_variables('vgg_16'))



from datasets import imagenet
imagenet.create_readable_names_for_imagenet_labels()



# Infer class and probability

with tf.Session() as sess:
    load_vars(sess)
    network_input, probabilities = sess.run([processed_images,
                                             probabilities])
    probabilities = probabilities[0, 0:]
    sorted_inds = [i[0] for i in sorted(enumerate(-probabilities),
                                        key=lambda x:x[1])]
    names_ = imagenet.create_readable_names_for_imagenet_labels()
    for i in range(5):
        index = sorted_inds[i]
        print('Class: ' + names_[index+1]
              + ' |prob: ' + str(probabilities[index]))




# Class: sports car, sport car |prob: 0.82706714
# Class: convertible |prob: 0.121342406
# Class: racer, race car, racing car |prob: 0.027330527
# Class: car wheel |prob: 0.016851021
# Class: grille, radiator grille |prob: 0.0058656675
