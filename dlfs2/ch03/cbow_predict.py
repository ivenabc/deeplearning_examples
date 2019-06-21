# coding: utf-8
import sys 
sys.path.append('..')

import numpy as np
from simple_cbow import SimpleCBOW
from utils.layers import MatMul
from utils.tools import create_contexts_target,preprocess,convert_one_hot

if __name__ == '__main__':
    text = 'You say goodbye and I say hello.'
    corpus, word_to_id, id_to_word = preprocess(text)
    # print(corpus)  # [0 1 2 3 4 1 5 6]
    # {'you': 0, 'say': 1, 'goodbye': 2, 'and': 3, 'i': 4, 'hello': 5, '.': 6}
    # print(word_to_id)
    # {0: 'you', 1: 'say', 2: 'goodbye', 3: 'and', 4: 'i', 5: 'hello', 6: '.'}
    # print(corpus[1:-1])  # [1 2 3 4 1 5]
    contexts, target = create_contexts_target(corpus)
    contexts = convert_one_hot(contexts, len(word_to_id))
    target = convert_one_hot(target, len(word_to_id))
    # print(contexts) [[0 2][1 3][2 4][3 1][4 5][1 6]]
    # print(contexts[:, 1]) [2 3 4 1 5 6]
    print(contexts.shape)
    print(np.dot(contexts[:,1], 0.01 * np.random.randn(len(word_to_id), 5).astype('f')))
    # model = SimpleCBOW(vocab_size, hidden_size)
    # model.forward(contexts, target)
