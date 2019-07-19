#coding:utf-8
import sys,os 
sys.path.append(os.pardir)
from dataset.mnist import load_mnist
from PIL import Image
import numpy as np 

def img_show(img):
    
    pil_img = Image.fromarray(np.uint8(img))
    pil_img.show()
    pil_img.save('./tmp.png')


(x_train,t_train),(x_test,t_test) = load_mnist(flatten=True, normalize=False)
print(x_train.shape) #(60000, 784)

img = x_train[0].reshape(28,28)
img_show(img)

