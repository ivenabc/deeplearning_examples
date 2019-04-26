import numpy as np 



def gradient_descent(f, init_x, lr=0.01, step_num=100):
    x = init_x
    for i in range(step_num):
        grad = numerical_gradient(f, x)
        x -= lr * grad
    return x


# 
def numerical_gradient(f, x):
    h = 1e-4 # 0.0001
    grad = np.zeros_like(x) # 生成和x形状相同的数组
    for idx in range(x.size):
        tmp_val = x[idx]
        # f(x+h)的计算
        x[idx] = tmp_val + h
        fxh1 = f(x)
        # f(x-h)的计算
        x[idx] = tmp_val - h
        fxh2 = f(x)
        grad[idx] = (fxh1 - fxh2) / (2*h)
        x[idx] = tmp_val # 还原值
    return grad


# 问题：请用梯度法求 的最小值。
# >>> def function_2(x):
# ... return x[0]**2 + x[1]**2
# ...
# >>> init_x = np.array([-3.0, 4.0])
# >>> gradient_descent(function_2, init_x=init_x, lr=0.1, step_num=100)
# array([ -6.11110793e-10, 8.14814391e-10])