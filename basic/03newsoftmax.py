import numpy as np 

a = np.array([0.3, 2.9, 4.0])

exp_a = np.exp(a)

c = np.max(a)

print(exp_a)
print(a-c)

def softmax(a):
    c = np.max(a)
    exp_a = np.exp(a - c) # 溢出对策
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a
    return y