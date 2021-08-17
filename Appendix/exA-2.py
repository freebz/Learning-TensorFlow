# 모듈형 설계

import tensorflow as tf
import numpy as np

x_data = np.random.randn(2000,3)
w_real = [0.3,0.5,0.1]
b_real = -0.2

noise = np.random.randn(1,2000) * 0.1
y_data = np.matmul(w_real, x_data.T) + b_real + noise

NUM_STEPS = 10

def predict(x,y_true,w,b):
    y_pred = tf.matmul(w, tf.transpose(x)) + b
    return y_pred

def get_loss(y_pred,y_true):
    loss = tf.reduce_mean(tf.square(y_true-y_pred))
    return loss

def get_optimizer(y_pred,y_true):
    learning_rate = 0.5
    loss = get_loss(y_pred,y_true)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    train = optimizer.minimize(loss)
    return train

def run_model(x_data,y_data):
    wb_ = []
    # 플레이스홀더와 변수 정의
    x = tf.placeholder(tf.float32, shape=[None,3])
    y_true = tf.placeholder(tf.float32, shape=None)
    w = tf.Variable([[0,0,0]], dtype=tf.float32)
    b = tf.Variable(0, dtype=tf.float32)
    print(b.name)
    
    # 예측 수행
    y_pred = predict(x,y_true,w,b)

    # 최적화 함수 생성
    train = get_optimizer(y_pred,y_data)

    # 세션 실행
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        for step in range(NUM_STEPS):
            sess.run(train, {x: x_data, y_true: y_data})
            if (step % 5 == 4) or (step == 0):
                print(step, sess.run([w,b]))
                wb_.append(sess.run([w,b]))

            print(10, sess.run([w,b]))

run_model(x_data,y_data)
run_model(x_data,y_data)



# Variable_5:0
# 0 [array([[0.29813892, 0.5288865 , 0.08942644]], dtype=float32), -0.24078882]
# 4 [array([[0.29807445, 0.49817857, 0.09742127]], dtype=float32), -0.20010309]
# 9 [array([[0.29807514, 0.4981774 , 0.097422  ]], dtype=float32), -0.20010057]
# Variable_7:0
# 0 [array([[0.29813892, 0.5288865 , 0.08942644]], dtype=float32), -0.24078882]
# 4 [array([[0.29807445, 0.49817857, 0.09742127]], dtype=float32), -0.20010309]
# 9 [array([[0.29807514, 0.4981774 , 0.097422  ]], dtype=float32), -0.20010057]
