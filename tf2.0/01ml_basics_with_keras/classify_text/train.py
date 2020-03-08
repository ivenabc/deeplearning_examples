#coding:utf-8
import os 
import numpy as np
import tensorflow as tf
from tensorflow import keras
from model import create_model

print("Version: ", tf.__version__)
# print("GPU is", "available" if tf.config.experimental.list_physical_devices("GPU") else "NOT AVAILABLE")

vocab_size = 10000

model = create_model(vocab_size)
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


imdb = tf.keras.datasets.imdb
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(
    path=os.path.abspath('./imdb/imdb.npz'),
    num_words=10000,
)

#Training entries: 25000, labels: 25000
print("Training entries: {}, labels: {}".format(len(train_data), len(train_labels))) 

# 由于每条数据长度不一致，所以需要做padding处理，不足部分补0


# 一个映射单词到整数索引的词典
word_index = imdb.get_word_index()

# 保留第一个索引
word_index = {k:(v+3) for k,v in word_index.items()}
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2  # unknown
word_index["<UNUSED>"] = 3

train_data = keras.preprocessing.sequence.pad_sequences(train_data,
                                                        value=word_index["<PAD>"],
                                                        padding='post',
                                                        maxlen=256)

test_data = keras.preprocessing.sequence.pad_sequences(test_data,
                                                       value=word_index["<PAD>"],
                                                       padding='post',
                                                       maxlen=256)

print(train_data.shape, train_labels.shape) #(25000, 256) (25000,)



x_val = train_data[:10000]
partial_x_train = train_data[10000:]

y_val = train_labels[:10000]
partial_y_train = train_labels[10000:]

history = model.fit(partial_x_train,
                    partial_y_train,
                    epochs=40,
                    callbacks=[cp_callback],
                    batch_size=512,
                    validation_data=(x_val, y_val),
                    verbose=1)

results = model.evaluate(test_data,  test_labels, verbose=2)

print(results)

# history_dict = history.history
# history_dict.keys()