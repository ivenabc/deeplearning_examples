import sys, os
sys.path.append('../../')
import numpy as np
from common.functions import softmax, cross_entropy_error 
from common.gradient import numerical_gradient

class simpleNet:
    def __init__(self):
        self.W = np.random.randn(2,3) # 用高斯分布进行初始化 def predict(self, x):

    def predict(self, x):
        return np.dot(x, self.W)

    def loss(self, x, t):
        z = self.predict(x)
        y = softmax(z)
        loss = cross_entropy_error(y, t)
        print('loss', loss)
        return loss


def gradient_descent(f, init_x, lr=0.01, step_num=100):
    x = init_x
    for i in range(step_num):
        grad = numerical_gradient(f, x) 
        x -= lr * grad
    return x


if __name__ == '__main__':
    net = simpleNet()
    print(net.W)
    x = np.array([0.6, 0.9])
    # p = net.predict(x)
    # print(p)
    t = np.array([0, 0, 1])
    # loss = net.loss(x, t)
    # print(loss)
    f = lambda W: net.loss(x, t)
    # numerical_gradient(f, net.W)
    grad = net.W
    gradient_descent(f, grad, lr=0.1, step_num=1000)
    
    

