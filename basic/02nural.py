import numpy as np 
import matplotlib.pylab as plt


# 3层神经网络：输入层（第0层）有2个神经元，第1个隐藏层（第1层）有3个神经元，
# 第2个隐藏层（第2层）有2个神经元，输出层（第3层）有2个神经元

X = np.array([1.0, 0.5]) # 1x2

W1 = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]]) #2x3