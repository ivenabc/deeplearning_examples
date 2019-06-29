import numpy as np 

x = np.array([
    [1001, 1002],
    [3, 4]
])

max_x = np.max(x, axis=1)
print('keepdims=false:',max_x) #[1002    4]
print('keepdims=false, result:', x-max_x) # [[  -1  998] [-999    0]]
print('x ===>', x.size)

keepdims_max_x = np.max(x, axis=1, keepdims=True)
print('keepdims=true:', keepdims_max_x) # [[1002] [   4]]
print('keepdims=true, result:', x-keepdims_max_x) # [[-1  0] [-1  0]]

