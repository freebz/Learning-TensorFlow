# 손실함수와 가중치를 얻어오는 함수 추가

import functools

def property_with_check(input_fn):
    attribute = '_cache_' + input_fn.__name__

    @property
    @functools.wraps(input_fn)
    def check_attr(self):
        if not hasattr(self, attribute):
            setattr(self, attribute, input_fn(self))
        return getattr(self, attribute)

    return check_attr



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

        self.params = self._initialize_weights()

        init = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(init)

        self.output
        self.optimizer
        self.loss


    def _initialize_weights(self):
        params = dict()
        params['w'] = tf.Variable([[0,0,0]],dtype=tf.float32)
        params['b'] = tf.Variable(0,dtype=tf.float32)
        return params

    def fit(self,x_data,y_data):
        print(self.params['b'].name)

        for step in range(10):
            self.sess.run(self.optimizer,{self.x: x_data, self.y_true: y_data})
            if (step % 5 == 4) or (step == 0):
                print(step, self.sess.run([self.params['w'],self.params['b']]))

    def evaluate(self,x_data,y_data):
        print(self.params['b'].name)

        MSE = self.sess.run(self.loss,{self.x: x_data, self.y_true: y_data})
        return MSE

    def getWeights(self):
        return self.sess.run([self.params['b']])
    

    @property_with_check
    def output(self):
        y_pred = tf.matmul(self.params['w'],tf.transpose(self.x)) + \
            self.params['b']
        return y_pred

    @property_with_check
    def loss(self):
        error = tf.reduce_mean(tf.square(self.y_true-self.output))
        return error

    @property_with_check
    def optimizer(self):
        opt = tf.train.GradientDescentOptimizer(0.5)
        opt = opt.minimize(self.loss)
        return opt
    

lin_reg = Model()
lin_reg.fit(x_data,y_data)
lin_reg.fit(x_data,y_data)
MSE = lin_reg.evaluate(x_data,y_data)
print(MSE)

print(lin_reg.getWeights())



# Variable_1:0
# 0 [array([[0.28772098, 0.46599895, 0.09041754]], dtype=float32), -0.19745615]
# 4 [array([[0.3029584 , 0.5003109 , 0.09881483]], dtype=float32), -0.19930114]
# 9 [array([[0.30295852, 0.5003118 , 0.09881495]], dtype=float32), -0.19930102]
# Variable_1:0
# 0 [array([[0.30295852, 0.5003118 , 0.09881495]], dtype=float32), -0.19930102]
# 4 [array([[0.30295852, 0.5003118 , 0.09881495]], dtype=float32), -0.19930102]
# 9 [array([[0.30295852, 0.5003118 , 0.09881495]], dtype=float32), -0.19930102]
# Variable_1:0
# 0.010578896
# [-0.19930102]
