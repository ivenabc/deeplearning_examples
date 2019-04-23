import tensorflow as tf
from tensorflow import keras
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt 
import numpy as np
import os

print(tf.__version__)

kerasPath = os.path.join(os.path.dirname(os.getcwd()), "data", "imdb", 'imdb.npz')

(train_data, train_labels), (test_data, test_labels) = keras.datasets.imdb.load_data(num_words=1000)

print(isinstance(train_data, np.ndarray))

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
model.add(keras.layers.Dense(16, activation='relu', input_shape=(10000,))) 
model.add(keras.layers.Dense(16, activation='relu')) 
model.add(keras.layers.Dense(1, activation='sigmoid'))

model.compile(optimizer='rmsprop',
    loss='binary_crossentropy', metrics=['accuracy'])

# 配置优化器
model.compile(optimizer=keras.optimizers.RMSprop(lr=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy'])

# 配置损失函数
model.compile(optimizer=keras.optimizers.RMSprop(lr=0.001),
    loss=keras.losses.binary_crossentropy,
    metrics=[keras.metrics.binary_accuracy])

x_val = x_train[:10000]
partial_x_train = x_train[10000:20000]
y_val = y_train[:10000] 
partial_y_train = y_train[10000:20000]

model.compile(optimizer='rmsprop',
    loss='binary_crossentropy',
    metrics=['acc'])

history = model.fit(partial_x_train, partial_y_train,
    epochs=20,
    batch_size=512, validation_data=(x_val, y_val))

history_dict = history.history

loss_value = history_dict['loss']
val_loss_value = history_dict['val_loss']

print(loss_value)
print(val_loss_value)
epochs = range(1, len(loss_value) + 1)

plt.plot(epochs, loss_value, 'bo', label='Training loss') 
plt.plot(epochs, val_loss_value, 'b', label='Validation loss') 
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss') 
plt.legend()
plt.savefig('plt.png')
# history_dict.keys()

# 层：深度学习的基础组件
# 层是一个数据处理模块，将一个
# 或多个输入张量转换为一个或多个输出张量。有些层是无状态的，但大多数的层是有状态的，
# 即层的权重。权重是利用随机梯度下降学到的一个或多个张量，其中包含网络的知识。
# 不同的张量格式与不同的数据处理类型需要用到不同的层。例如，简单的向量数据保存在
# 形状为 (samples, features) 的 2D 张量中，通常用密集连接层［densely connected layer，也
# 叫全连接层（fully connected layer）或密集层（dense layer），对应于 Keras 的 Dense 类］来处
# 理。序列数据保存在形状为 (samples, timesteps, features) 的 3D 张量中，通常用循环
# 层（recurrent layer，比如 Keras 的 LSTM 层）来处理。图像数据保存在 4D 张量中，通常用二维
# 卷积层（Keras 的 Conv2D ）来处理。
# layer = layers.Dense(32, input_shape=(784,)) 有 32 个输出单元的密集层