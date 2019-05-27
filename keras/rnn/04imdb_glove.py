import os 
import tensorflow as tf 
import numpy as np 

imdb_dir = '../../data/'
train_dir = os.path.join(imdb_dir, 'prodimdb')
glove_dir = os.path.join(imdb_dir, 'glove.6B/glove.6B.100d.txt')
print(train_dir)

labels = []
texts = []

for label_type in ['neg', 'pos']:
    dir_name = os.path.join(train_dir, label_type)
    for fname in os.listdir(dir_name):
        if fname[-4:] == '.txt':
            f = open(os.path.join(dir_name, fname), encoding='UTF-8')
            text = f.read()
            texts.append(text)
            f.close()
            if label_type == 'neg':
                labels.append(0)
            else:
                labels.append(1)



maxlen = 100 # 在100个单词后截断评论
training_samples = 200 # 在200个样本上训练
validation_samples = 10000 #
max_words = 10000 #只考虑数据集中前10 000个最常见的单词

tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(texts)
# 获取数据字典 
# eg: {'the': 1, 'and': 2, 'a': 3, 'to': 4}
sequences = tokenizer.texts_to_sequences(texts)

word_index = tokenizer.word_index
# print('Found %s unique tokens.' % len(word_index))
# print('Found  ', word_index.get('again'))
# print('Found :', word_index)
data = tf.keras.preprocessing.sequence.pad_sequences(sequences, maxlen=maxlen)


# new_dict = {v:k for k, v in word_index.items()}
# print('Found new dict:', new_dict[81], new_dict[42])
# print('Found data shape:',data.shape)
# print('Found data:', data[0])


# 打乱数组顺序
# indices = np.arange(data.shape[0])
# print(indices)
# np.random.shuffle(indices)
# print(indices)
# data = data[indices]
# labels = labels[indices]
# [ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16]
# [ 6 16 15  9 12  4 14  0  2 10  3 11 13  1  7  8  5]

x_train = data[:training_samples]
y_train = labels[:training_samples]
x_val = data[training_samples: training_samples + validation_samples]
y_val = labels[training_samples: training_samples + validation_samples]

embeddings_index =  {}

f = open(glove_dir, encoding='UTF-8')
for line in f:
    values = line.split()
    # print('values:', values)
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

print('Found %s word vectors.' % len(embeddings_index))

embedding_dim = 100
embedding_matrix = np.zeros((max_words, embedding_dim))

for word, i in word_index.items():
    if i < max_words:
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector


model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Embedding(max_words, embedding_dim,input_length=maxlen))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(32, activation='relu'))
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
model.summary()
model.layers[0].set_weights([embedding_matrix])
model.layers[0].trainable = False
model.compile(
    optimizer='rmsprop',
    loss='binary_crossentropy',
    metrics=['acc'])
history = model.fit(
    x_train, 
    y_train,
    epochs=10,
    batch_size=32,
    validation_data=(x_val, y_val))
model.save_weights('pre_trained_glove_model.h5')