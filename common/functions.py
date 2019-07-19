import numpy as np 

def sotmax(x):
    if x.ndim == 2:
        x = x.T 
        x = x - np.max(x, axis=0)
        y = np.exp(x) / np.sum(np.exp(x), axis=0)
        print('sum:',np.sum(np.exp(x), axis=0))
        return y.T
    x = x - np.max(x)
    return np.exp(x)/np.sum(np.exp(x))


if __name__ == '__main__':
    x = np.array([
        [1,2,3,4],
        [4,5,6,8]
    ])

    y = sotmax(x)
    print('y:', y)