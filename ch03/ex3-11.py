import tensorflow as tf

A = tf.constant([ [1,2,3],
                  [4,5,6] ])
print(A.get_shape())

x = tf.constant([1,0,1])
print(x.get_shape())

# (2, 3)
# (3,)
