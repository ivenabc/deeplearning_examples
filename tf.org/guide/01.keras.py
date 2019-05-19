import tensorflow as tf
from tensorflow.keras import layers
import os 
import numpy as np

keras = tf.keras

print(tf.VERSION)
print(tf.keras.__version__)

kerasPath = os.path.join(os.path.dirname("../../"), "data", "imdb", 'imdb.npz')

(train_data, train_labels), (test_data, test_labels) = tf.keras.datasets.imdb.load_data(num_words=1000)


def vectorize_sequences(sequences, dimension=10000): 
    # 建一个形状为 (len(sequences), dimension) 的零矩阵
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        #  将 results[i] 的指定索引设为 1 
        results[i, sequence] = 1. 
    return results

x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)

y_train = np.asarray(train_labels).astype('float32')
y_test = np.asarray(test_labels).astype('float32')

model = keras.models.Sequential()
model.add(keras.layers.Dense(16, activation=tf.nn.relu, input_shape=(10000,))) 
model.add(keras.layers.Dense(16, activation=tf.nn.relu)) 
model.add(keras.layers.Dense(1, activation=tf.nn.sigmoid))

x_val = x_train[:10000]
partial_x_train = x_train[10000:20000]
y_val = y_train[:10000] 
partial_y_train = y_train[10000:20000]


model.compile(optimizer=tf.train.RMSPropOptimizer(0.001),
    loss='binary_crossentropy', metrics=['accuracy'])

# tensorboard --logdir logs/
tbCallBack = tf.keras.callbacks.TensorBoard(log_dir='./logs',histogram_freq=0, write_graph=True, write_images=True)

model.fit(partial_x_train, partial_y_train, epochs=20,  
    batch_size=512, validation_data=(x_val, y_val),
    callbacks=[tbCallBack])


# model.save_weights('./weights/my_model')

# class MyModel(tf.keras.Model):
#     def init(self):
#         self.dense1 = keras.layers.Dense(16, activation=tf.nn.relu, input_shape=(10000,))
#         self.dense2 = keras.layers.Dense(16, activation=tf.nn.relu)   