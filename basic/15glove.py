from functools import wraps
from collections import Counter

test_corpus = ("""human interface computer
survey user computer system response time
eps user interface system
system human system eps
user response time
trees
graph trees
graph minors trees
graph minors survey
I like graph and stuff
I like trees and stuff
Sometimes I build a graph
Sometimes I build trees""").split("\n")


# https://github.com/hans/glove.py/blob/master/util.py
# http://www.foldl.me/2014/glove-python/

def listify(fn):
    """
    Use this decorator on a generator function to make it return a list
    instead.
    """

    @wraps(fn)
    def listified(*args, **kwargs):
        return list(fn(*args, **kwargs))

    return listified


def build_vocab(corpus):
    vocab = Counter()
    for line in corpus:
        tokens = line.strip().split()
        vocab.update(tokens)
    print(vocab)
    return {word: (i, freq) for i, (word, freq) in enumerate(vocab.items())}

if __name__ == '__main__':
    print(build_vocab(test_corpus))
    
# glove.logger.setLevel(logging.ERROR)
# vocab = glove.build_vocab(test_corpus)
# cooccur = glove.build_cooccur(vocab, test_corpus, window_size=10)
# id2word = evaluate.make_id2word(vocab)

# W = glove.train_glove(vocab, cooccur, vector_size=10, iterations=500)

# # Merge and normalize word vectors
# W = evaluate.merge_main_context(W)


# def test_similarity():
#     similar = evaluate.most_similar(W, vocab, id2word, 'graph')
#     logging.debug(similar)

#     assert_equal('trees', similar[0])