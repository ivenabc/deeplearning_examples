import numpy as np
import sys, os
sys.path.append(os.pardir)
from common.functions import *
from common.gradient import numerical_gradient
from dataset.mnist import load_mnist

# 步骤1（mini-batch）
# 从训练数据中随机选出一部分数据，这部分数据称为mini-batch。我们
# 的目标是减小mini-batch的损失函数的值。
# 步骤2（计算梯度）
# 为了减小mini-batch的损失函数的值，需要求出各个权重参数的梯度。
# 梯度表示损失函数的值减小最多的方向。
# 步骤3（更新参数）
# 将权重参数沿梯度方向进行微小更新。
# 步骤4（重复）
# 重复步骤1、步骤2、步骤3。

# epoch 是一个单位。一个 epoch 表示学习中所有训练数据均被使用过 一次时的更新次数。
# 比如，对于 10000 笔训练数据，用大小为 100 笔数据的 mini-batch 进行学习时，
# 重复随机梯度下降法 100 次，所 有的训练数据就都被“看过”了 A。此时，
# 100 次就是一个 epoch。


class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        self.params = {}
        self.params['W1'] = weight_init_std *  np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size) 
        self.params['b2'] = np.zeros(output_size)
    
    def predict(self, x):
        W1, W2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.params['b2']

        a1 = np.dot(x, W1) + b1
        z1 = sigmoid(a1)

        a2 = np.dot(z1, W2) + b2
        y = softmax(a2)
        return y
    
    def loss(self, x, t):
        y = self.predict(x)
        return cross_entropy_error(y, t)

    def numerical_gradient(self, x, t):
        loss_W = lambda W: self.loss(x, t)
        grads = {}
        grads['W1'] = numerical_gradient(loss_W, self.params['W1']) 
        grads['b1'] = numerical_gradient(loss_W, self.params['b1']) 
        grads['W2'] = numerical_gradient(loss_W, self.params['W2']) 
        grads['b2'] = numerical_gradient(loss_W, self.params['b2'])
        return grads


(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

train_loss_list = []

iters_num = 10000
train_size = x_train.shape[0] 
batch_size = 100 
learning_rate = 0.1
network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

for i in range(iters_num):
    print("current:", i)
    batch_mask = np.random.choice(train_size, batch_size) 
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]
    grad = network.numerical_gradient(x_batch, t_batch)

    for key in ('W1', 'b1', 'W2', 'b2'):
        network.params[key] -= learning_rate * grad[key]
    
    loss = network.loss(x_batch, t_batch)
    train_loss_list.append(loss)
    print("train loss:", loss)

