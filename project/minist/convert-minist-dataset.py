from PIL import Image
import tensorflow as tf 
import numpy as np
import matplotlib.pyplot as plt
import os


(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

for i in range(10):
    os.makedirs('./train/%s' % i, exist_ok=True)
    os.makedirs('./test/%s' % i,exist_ok=True)

for i in range(len(train_images)):
    im = Image.fromarray(train_images[i])
    im.save('./train/%s/train_%s.bmp' % (train_labels[i], i),'bmp')

for i in range(len(test_images)):
    im = Image.fromarray(test_images[i])
    im.save('./test/%s/test_%s.bmp' % (test_labels[i], i),'bmp')

# plt.figure()
# plt.imshow(train_images[0])
# plt.colorbar()
# plt.grid(False)