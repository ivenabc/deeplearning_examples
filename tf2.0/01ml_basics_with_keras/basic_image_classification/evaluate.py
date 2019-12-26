import os 
from load_data import load_mnist
from model import create_model
import tensorflow as tf 

test_images, test_labels = load_mnist('./fashion', kind='t10k')
model = create_model()

checkpoint_path = 'training_models/cp-{epoch:04d}.ckpt'
checkpoint_dir = os.path.dirname(checkpoint_path)
latest = tf.train.latest_checkpoint(checkpoint_dir)
if latest:
    model.load_weights(latest)
    
test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
print('\nTest accuracy:', test_acc)
print('\nTest loss:', test_loss)