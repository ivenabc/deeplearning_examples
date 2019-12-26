import os 
from load_data import load_mnist
from model import create_model
import numpy as np 
import tensorflow as tf 

test_images, test_labels = load_mnist('./fashion', kind='t10k')
model = create_model()

checkpoint_path = 'training_models/cp-{epoch:04d}.ckpt'
checkpoint_dir = os.path.dirname(checkpoint_path)
latest = tf.train.latest_checkpoint(checkpoint_dir)
if latest:
    model.load_weights(latest)

img = test_images[1]
img = (np.expand_dims(img,0))
predictions_single = model.predict(img)

print(predictions_single)
print(np.argmax(predictions_single[0]))