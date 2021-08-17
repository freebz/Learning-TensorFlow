import tensorflow as tf

sess = tf.InteractiveSession()
queue1 = tf.FIFOQueue(capacity=10, dtypes=[tf.string])

enque_op = queue1.enqueue(["F"])

# size is 0 before run
sess.run(queue1.size())
# 0

enque_op.run()
sess.run(queue1.size())
# 1

enque_op = queue1.enqueue(["I"])
enque_op.run()
enque_op = queue1.enqueue(["F"])
enque_op.run()
enque_op = queue1.enqueue(["O"])
enque_op.run()

sess.run(queue1.size())
# 4


x = queue1.dequeue()
x.eval()
# b'F'
x.eval()
# b'I'
x.eval()
# b'F'
x.eval()
# b'O'


# for dequeue many, need to specify shapes in advance...
queue1 = tf.FIFOQueue(capacity=10, dtypes=[tf.string], shapes=[()])


enque_op = queue1.enqueue(["F"])
enque_op.run()
enque_op = queue1.enqueue(["I"])
enque_op.run()
enque_op = queue1.enqueue(["F"])
enque_op.run()
enque_op = queue1.enqueue(["O"])
enque_op.run()


inputs = queue1.dequeue_many(4)
inputs.eval()
# array([b'F', b'I', b'F', b'O'], dtype=object)




# Multithreading
# single queue, but execute sess.run calls in parallel...

import threading

gen_random_normal = tf.random_normal(shape=())
queue = tf.FIFOQueue(capacity=100, dtypes=[tf.float32], shapes=())
enque = queue.enqueue(gen_random_normal)

def add():
    for i in range(10):
        sess.run(enque)


# Create 10 threads that run add()
threads = [threading.Thread(target=add, args=()) for i in range(10)]

threads
# [<Thread(Thread-1, initial)>,
#  <Thread(Thread-2, initial)>,
#  <Thread(Thread-3, initial)>,
#  <Thread(Thread-4, initial)>,
#  <Thread(Thread-5, initial)>,
#  <Thread(Thread-6, initial)>,
#  <Thread(Thread-7, initial)>,
#  <Thread(Thread-8, initial)>,
#  <Thread(Thread-9, initial)>,
#  <Thread(Thread-10, initial)>]


import time

for t in threads:
    t.start()

print(sess.run(queue.size()))
time.sleep(0.01)
print(sess.run(queue.size()))
time.sleep(0.01)
print(sess.run(queue.size()))


# 10
# 84
# 100


x = queue.dequeue_many(10)
print(x.eval())
sess.run(queue.size())


# [ 0.21280588  1.08403     1.5146887  -0.10597554  0.6534186  -1.1490353
#  -0.5095637   0.21179083  0.12201982 -2.115584  ]
# 90





# A coordinator for threads.
# a simple mechanism to coordinate the termination of a set of threads

gen_random_normal = tf.random_normal(shape=())
queue = tf.FIFOQueue(capacity=100, dtypes=[tf.float32], shapes=())
enque = queue.enqueue(gen_random_normal)

def add(coord, i):
    while not coord.should_stop():
        sess.run(enque)
        if i == 11:
            coord.request_stop()

coord = tf.train.Coordinator()
threads = [threading.Thread(target=add, args=(coord, i)) for i in range(10)]
coord.join(threads)

for t in threads:
    t.start()

print(sess.run(queue.size()))
time.sleep(0.01)
print(sess.run(queue.size()))
time.sleep(0.01)
print(sess.run(queue.size()))


# 10
# 100
# 100



gen_random_normal = tf.random_normal(shape=())
queue = tf.FIFOQueue(capacity=100, dtypes=[tf.float32], shapes=())
enque = queue.enqueue(gen_random_normal)

def add(coord, i):
    while not coord.should_stop():
        sess.run(enque)
        if i == 1:
            coord.request_stop()

coord = tf.train.Coordinator()
threads = [threading.Thread(target=add, args=(coord, i)) for i in range(10)]
coord.join(threads)

for t in threads:
    t.start()

print(sess.run(queue.size()))
time.sleep(0.01)
print(sess.run(queue.size()))
time.sleep(0.01)
print(sess.run(queue.size()))


# 10
# 17
# 17




# tf.train.QueueRunner와 th.RandomShuffleQueue

gen_random_normal = tf.random_normal(shape=())
queue = tf.RandomShuffleQueue(capacity=100, dtypes=[tf.float32],
                              min_after_dequeue=1)
enqueue_op = queue.enqueue(gen_random_normal)

qr = tf.train.QueueRunner(queue, [enqueue_op] * 4)
coord = tf.train.Coordinator()
enqueue_threads = qr.create_threads(sess, coord=coord, start=True)
coord.request_stop()
coord.join(enqueue_threads)

print(sess.run(queue.size()))
