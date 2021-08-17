import tensorflow as tf

init_val = tf.random_normal((1,5),0,1)
var = tf.Variable(init_val, name='var')
print("pre run: \n{}".format(var))

init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    post_var = sess.run(var)

print("\npost run: \n{}".format(post_var))

# pre run: 
# <tf.Variable 'var:0' shape=(1, 5) dtype=float32_ref>

# post run: 
# [[-0.6226208   0.91896003 -0.7395614  -0.02814943  1.9727764 ]]


# pre run: 
# <tf.Variable 'var_1:0' shape=(1, 5) dtype=float32_ref>
