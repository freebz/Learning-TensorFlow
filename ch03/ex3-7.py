import tensorflow as tf

c = tf.constant(4.0, dtype=tf.float64)
print(c)
print(c.dtype)

# Tensor("Const_10:0", shape=(), dtype=float64)
# <dtype: 'float64'>
