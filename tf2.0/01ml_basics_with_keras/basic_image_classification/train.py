import os 
from load_data import load_mnist
from model import create_model
import tensorflow as tf 

train_images, train_labels = load_mnist('./fashion', kind='train')
test_images, test_labels = load_mnist('./fashion', kind='t10k')
model = create_model()

print(type(train_images),train_images.shape) # <class 'numpy.ndarray'> , shape: <class 'numpy.ndarray'> (60000, 28, 28)
print(type(train_labels),train_labels.shape, train_labels[0]) # <class 'numpy.ndarray'> (60000,) 9

checkpoint_path = 'training_models/cp-{epoch:04d}.ckpt'
cp_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path, 
    verbose=1, 
    save_weights_only=True,
    period=1)

checkpoint_dir = os.path.dirname(checkpoint_path)
latest = tf.train.latest_checkpoint(checkpoint_dir)
if latest:
    model.load_weights(latest)
    
model.fit(train_images, 
    train_labels, 
    epochs=10,
    callbacks=[cp_callback],
    validation_data=(test_images,test_labels),
    verbose=1)