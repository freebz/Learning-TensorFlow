import tensorflow as tf

with tf.Session() as sess:
    fetches = [a,b,c,d,e,f]
    outs = sess.run(fetches)

print("outs = {}".format(outs))
print(type(outs[0]))

# outs = [5, 2, 3, 10, 5, 5]
# <class 'numpy.int32'>
