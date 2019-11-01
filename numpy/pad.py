import sys, os
sys.path.append(os.pardir)
import numpy as np 
from common.utils import im2col


# arr1D = np.array([1, 1, 2, 2, 3, 7])

# print('linear_ramp:  ' + str(np.pad(arr1D, (2, 3), 'linear_ramp')))

x1 = np.random.rand(1, 3, 7, 7)
col1 = im2col(x1, 5, 5, stride=1, pad=0)
print(col1.shape) # (9, 75)
x2 = np.random.rand(10, 3, 7, 7) # 10个数据
col2 = im2col(x2, 5, 5, stride=1, pad=0)
print(col2.shape) # (90, 75)