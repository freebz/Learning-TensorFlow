import numpy as np

# 데이터를 생성하고 결과를 시뮬레이션
x_data = np.random.randn(2000,3)
w_real = [0.3,0.5,0.1]
b_real = -0.2

noise = np.random.randn(1,2000) * 0.1
y_data = np.matmul (w_real, x_data.T) + b_real + noise
