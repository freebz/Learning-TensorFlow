# 예제 2: 로지스틱 회귀

import numpy as np
import tensorflow as tf

N = 20000
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# 데이터를 생성하고 결과를 시뮬레이션
x_data = np.random.randn(N,3)
w_real = [0.3,0.5,0.1]
b_real = -0.2
wxb = np.matmul(w_real,x_data.T) + b_real

y_data_pre_noise = sigmoid(wxb)
y_data = np.random.binomial(1,y_data_pre_noise)


# y_pred = tf.sigmoid(y_pred)
# loss = -y_true*tf.log(y_pred) - (1-y_true)*tf.log(1-y_pred)
# loss = tf.reduce_mean(loss)


NUM_STEPS = 50

g = tf.Graph()
wb_ = []
with g.as_default():
    x = tf.placeholder(tf.float32,shape=[None,3])
    y_true = tf.placeholder(tf.float32,shape=None)

    with tf.name_scope('inference') as scope:
        w = tf.Variable([[0,0,0]],dtype=tf.float32,name='weights')
        b = tf.Variable(0,dtype=tf.float32,name='bias')
        y_pred = tf.matmul(w,tf.transpose(x)) + b

    with tf.name_scope('loss') as scope:
        loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=y_true,logits=y_pred)
        loss = tf.reduce_mean(loss)

    with tf.name_scope('train') as scope:
        learning_rate = 0.5
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        train = optimizer.minimize(loss)    

    # 시작하기 전에 변수를 초기화한다.
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        for step in range(NUM_STEPS):
            sess.run(train,{x: x_data, y_true: y_data})
            if (step % 5 == 0):
                print(step, sess.run([w,b]))
                wb_.append(sess.run([w,b]))

        print(50, sess.run([w,b]))


# 0 [array([[0.03151434, 0.05997963, 0.01104363]], dtype=float32), -0.021749988]
# 5 [array([[0.14047861, 0.26568365, 0.04888416]], dtype=float32), -0.096476965]
# 10 [array([[0.19967087, 0.3758852 , 0.06909329]], dtype=float32), -0.1365457]
# 15 [array([[0.23323876, 0.4376243 , 0.08036158]], dtype=float32), -0.15894893]
# 20 [array([[0.2528177 , 0.47325528, 0.08682945]], dtype=float32), -0.17183353]
# 25 [array([[0.2644281 , 0.4941899 , 0.09060847]], dtype=float32), -0.1793733]
# 30 [array([[0.2713794 , 0.5066218 , 0.09284051]], dtype=float32), -0.1838324]
# 35 [array([[0.27556416, 0.5140518 , 0.09416769]], dtype=float32), -0.18648693]
# 40 [array([[0.2780911 , 0.51850957, 0.09496016]], dtype=float32), -0.18807374]
# 45 [array([[0.27961954, 0.5211904 , 0.09543464]], dtype=float32), -0.18902482]
# 50 [array([[0.28039497, 0.5225439 , 0.0956733 ]], dtype=float32), -0.18950367]
