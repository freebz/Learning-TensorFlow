import tensorflow as tf

x = tf.expand_dims(x,1)
print(x.get_shape())

b = tf.matmul(A,x)

sess = tf.InteractiveSession()
print('matmul result:\n {}'.format(b.eval()))
sess.close()

# (3, 1)

# matmul result:
#  [[ 4]
#  [10]]
