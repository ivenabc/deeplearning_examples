#coding:utf-8
import sys,os 
sys.path.append(os.pardir)
from dataset.mnist import load_mnist
from PIL import Image
import numpy as np 


# 比如对于一个有 2000 个训练样本的数据集。将 2000 个样本分成大小为 500 的 batch，那么完成一个 epoch 需要 4 个 iteration。

def img_show(img):
    pil_img = Image.fromarray(np.uint8(img))
    pil_img.show()
    pil_img.save('./tmp.png')


(x_train,t_train),(x_test,t_test) = load_mnist(flatten=True, normalize=False)
print(x_train.shape) #(60000, 784)

img = x_train[0].reshape(28,28)
img_show(img)

