# 클래스 캡슐화

import tensorflow as tf
import numpy as np

x_data = np.random.randn(2000,3)
w_real = [0.3,0.5,0.1]
b_real = -0.2

noise = np.random.randn(1,2000) * 0.1
y_data = np.matmul(w_real, x_data.T) + b_real + noise


class Model:
    def __init__(self):

        # 모델
        self.x = tf.placeholder(tf.float32,shape=[None,3])
        self.y_true = tf.placeholder(tf.float32,shape=None)

        self.w = tf.Variable([[0,0,0]],dtype=tf.float32)
        self.b = tf.Variable(0,dtype=tf.float32)

        init = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(init)

        self._output = None
        self._optimizer = None
        self._loss = None

    def fit(self,x_data,y_data):
        print(self.b.name)

        for step in range(10):
            self.sess.run(self.optimizer,{self.x: x_data, self.y_true: y_data})
            if (step % 5 == 4) or (step == 0):
                print(step, self.sess.run([self.w,self.b]))


    @property
    def output(self):
        if not self._output:
            y_pred = tf.matmul(self.w,tf.transpose(self.x)) + self.b
            self._output = y_pred
        return self._output

    @property
    def loss(self):
        if not self._loss:
            error = tf.reduce_mean(tf.square(self.y_true-self.output))
            self._loss = error
        return self._loss

    @property
    def optimizer(self):
        if not self._optimizer:
            opt = tf.train.GradientDescentOptimizer(0.5)
            opt = opt.minimize(self.loss)
            self._optimizer = opt
        return self._optimizer

lin_reg = Model()
lin_reg.fit(x_data,y_data)
lin_reg.fit(x_data,y_data)



# Variable_1:0
# 0 [array([[0.31520292, 0.5161174 , 0.09528223]], dtype=float32), -0.18411334]
# 4 [array([[0.29773465, 0.5015807 , 0.10058893]], dtype=float32), -0.20033087]
# 9 [array([[0.2977344 , 0.50158054, 0.10058898]], dtype=float32), -0.20033106]
# Variable_1:0
# 0 [array([[0.2977344 , 0.50158054, 0.10058898]], dtype=float32), -0.20033105]
# 4 [array([[0.2977344 , 0.50158054, 0.10058898]], dtype=float32), -0.20033105]
# 9 [array([[0.2977344 , 0.50158054, 0.10058898]], dtype=float32), -0.20033106]
