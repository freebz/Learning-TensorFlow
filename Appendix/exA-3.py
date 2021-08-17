# 변수 공유

import tensorflow as tf
import numpy as np

x_data = np.random.randn(2000,3)
w_real = [0.3,0.5,0.1]
b_real = -0.2

noise = np.random.randn(1,2000) * 0.1
y_data = np.matmul(w_real, x_data.T) + b_real + noise

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
    x = tf.placeholder(tf.float32,shape=[None,3])
    y_true = tf.placeholder(tf.float32,shape=None)

    w = tf.get_variable('w',[1,3],initializer=tf.zeros_initializer())
    b = tf.get_variable('b',[1,1],initializer=tf.zeros_initializer())

    print(b.name,w.name)

    # 예측 수행
    y_pred = predict(x,y_true,w,b)

    # 최적화 함수 생성
    train = get_optimizer(y_pred,y_data)

    # 세션 실행
    init = tf.global_variables_initializer()
    sess.run(init)
    for step in range(10):
        sess.run(train,{x: x_data, y_true: y_data})
        if (step % 5 == 4) or (step == 0):
            print(step, sess.run([w,b]))
            wb_.append(sess.run([w,b]))

sess = tf.Session()

with tf.variable_scope("Regression") as scope:
    run_model(x_data,y_data)
    scope.reuse_variables()
    run_model(x_data,y_data)
sess.close()




# Regression/b:0 Regression/w:0
# 0 [array([[0.31520292, 0.5161174 , 0.09528223]], dtype=float32), array([[-0.18411334]], dtype=float32)]
# 4 [array([[0.29773465, 0.5015807 , 0.10058893]], dtype=float32), array([[-0.20033087]], dtype=float32)]
# 9 [array([[0.2977344 , 0.50158054, 0.10058898]], dtype=float32), array([[-0.20033106]], dtype=float32)]
# Regression/b:0 Regression/w:0
# 0 [array([[0.31520292, 0.5161174 , 0.09528223]], dtype=float32), array([[-0.18411334]], dtype=float32)]
# 4 [array([[0.29773465, 0.5015807 , 0.10058893]], dtype=float32), array([[-0.20033087]], dtype=float32)]
# 9 [array([[0.2977344 , 0.50158054, 0.10058898]], dtype=float32), array([[-0.20033106]], dtype=float32)]
