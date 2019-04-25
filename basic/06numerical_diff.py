import numpy as np 



# 利用微小的差分求导数的过程称为数值微分
def numerical_diff(f, x):
    h = 1e-4 # 0.0001
    return (f(x+h) - f(x-h)) / (2*h)

# f(X0, X1) = x0^2 + x1^2
# 有多个变量的函数的导数称为偏导数
def function_2(x):
    # return x[0]**2 + x[1]**2
    return np.sum(x**2)

def function_tmp1(x0):
    return x0*x0 + 4.0**2.0

# X0=,X1=4 关于x0得导数是
# numerical_diff(function_tmp1, 3.0)
# 6.00000000

# X0=3,X1=4 关于 X1得偏导数
def function_tmp2(x1):
    return 3.0**2.0 + x1*x1

# numerical_diff(function_tmp2, 4.0)
# numerical_diff(function_tmp2, 4.0)
# 7.999999999999119
