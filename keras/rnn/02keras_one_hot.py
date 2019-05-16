from tensorflow.keras.preprocessing.text import Tokenizer

# one-hot 散列技巧
# 单词散列编码为固定长度的向量
samples = ['The cat sat on the mat.', 'The dog ate my homework.']
tokenizer = Tokenizer(num_words=1000)
tokenizer.fit_on_texts(samples)
sequences = tokenizer.texts_to_sequences(samples)
one_hot_results = tokenizer.texts_to_matrix(samples, mode='binary')
word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))
print(one_hot_results.shape) #(2, 1000)