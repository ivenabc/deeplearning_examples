#coding:utf-8
import os 
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

print(tf.__version__)

def load_mnist(path, kind='train'):
    import os
    import gzip
    import numpy as np

    """Load MNIST data from `path`"""
    labels_path = os.path.join(path,
                               '%s-labels-idx1-ubyte.gz'
                               % kind)
    images_path = os.path.join(path,
                               '%s-images-idx3-ubyte.gz'
                               % kind)

    with gzip.open(labels_path, 'rb') as lbpath:
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8,
                               offset=8)

    with gzip.open(images_path, 'rb') as imgpath:
        images = np.frombuffer(imgpath.read(), dtype=np.uint8,
                               offset=16).reshape(len(labels), 28,28)

    return images, labels

train_images, train_labels = load_mnist('fashion', kind='train')
test_images, test_labels = load_mnist('fashion', kind='t10k')

print(type(train_images),train_images.shape) # <class 'numpy.ndarray'> , shape: <class 'numpy.ndarray'> (60000, 28, 28)
print(type(train_labels),train_labels.shape, train_labels[0]) # <class 'numpy.ndarray'> (60000,) 9

# plt.figure()
# plt.imshow(train_images[0])
# plt.colorbar()
# plt.grid(False)
# plt.show()

model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy'])

checkpoint_path = 'training_models/cp-{epoch:04d}.ckpt'
# checkpoint_dir = os.path.dirname(checkpoint_path)
cp_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path, 
    verbose=1, 
    save_weights_only=True,
    period=1)

checkpoint_dir = os.path.dirname(checkpoint_path)
latest = tf.train.latest_checkpoint(checkpoint_dir)
if latest:
    model.load_weights(latest)
# model.save_weights(checkpoint_path.format(epoch=0))

model.fit(train_images, 
    train_labels, 
    epochs=10,
    callbacks=[cp_callback],
    validation_data=(test_images,test_labels),
    verbose=1)

# test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)

# print('\nTest accuracy:', test_acc)

# predictions = model.predict(test_images)