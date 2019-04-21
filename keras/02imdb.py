from tensorflow import keras
import numpy as np
from tensorflow.keras.datasets import imdb
import os 

kerasPath = os.path.join(os.path.dirname(os.getcwd()), "data", "imdb", 'imdb.pkl')


(train_data, train_labels), (test_data, test_labels) = keras.datasets.imdb.load_data(path=kerasPath,num_words=10000)

# print(isinstance(train_data, np.ndarray))

def vectorize_sequences(sequences, dimension=10000): 
    # 建一个形状为 (len(sequences), dimension) 的零矩阵
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):’
        #  将 results[i] 的指定索引设为 1 
        results[i, sequence] = 1. 
        return results

print(train_data.shape)
print(train_labels.shape)
print(len(train_data[24999]))

model = keras.models.Sequential()
model.add(keras.layers.Dense(16, activation='relu', input_shape=(10000,))) 
model.add(keras.layers.Dense(16, activation='relu')) 
model.add(keras.layers.Dense(1, activation='sigmoid'))

model.compile(optimizer='rmsprop',
    loss='binary_crossentropy', metrics=['accuracy'])

# x_val = x_train[:10000]
# partial_x_train = x_train[10000:20000]
# y_val = y_train[:10000] 
# partial_y_train = y_train[10000:20000]

# history = model.fit(partial_x_train, partial_y_train,
#     epochs=20,
#     batch_size=512, validation_data=(x_val, y_val))

# history_dict = history.history
# history_dict.keys()

