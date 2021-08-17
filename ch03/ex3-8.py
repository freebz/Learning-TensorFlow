import tensorflow as tf

x = tf.constant([1,2,3],name='x',dtype=tf.float32)
print(x.dtype)
x = tf.cast(x,tf.int64)
print(x.dtype)

# <dtype: 'float32'>
# <dtype: 'int64'>
