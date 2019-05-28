import tensorflow as tf 
import os 
from tensorflow.keras import datasets, layers, models

train_files = []
train_labels = []

BATCH_SIZE = 32
print(tf.__version__)

for i in range(10):
    files = ['./train/%s/%s' % (i,j) for j in os.listdir('./train/%s' % i )]
    train_files.extend(files)
    train_labels.extend([ i for _ in files ])


# train_labels = tf.keras.utils.to_categorical(train_labels)
# test_labels = keras.utils.to_categorical(test_labels)

label_ds = tf.data.Dataset.from_tensor_slices(tf.cast(train_labels, tf.int32))
filepath_ds = tf.data.Dataset.from_tensor_slices(train_files)

ds = tf.data.Dataset.zip((filepath_ds, label_ds))

def preprocess_image(image):
    image = tf.image.decode_bmp(image, channels=1)
    image = tf.image.resize(image, [28,28])
    image /= 255.0
    return image

def load_and_preprocess_image(path):
    image = tf.io.read_file(path)
    return preprocess_image(image)

def load_and_preprocess_from_path_label(path, label):
    return load_and_preprocess_image(path), label

image_label_ds = ds.map(load_and_preprocess_from_path_label)
ds = image_label_ds.shuffle(buffer_size=len(train_files))
ds = ds.repeat()
ds = ds.batch(BATCH_SIZE)
ds = ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

# tensorboard --logdir logs/
tbCallBack = tf.keras.callbacks.TensorBoard(log_dir='./logs',histogram_freq=0, write_graph=True, write_images=True)

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))

model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy']
              )

steps_per_epoch=tf.math.ceil(len(train_files)/BATCH_SIZE).numpy()

model.fit(
    ds, 
    epochs=5, 
    callbacks=[tbCallBack],
    steps_per_epoch=steps_per_epoch
)