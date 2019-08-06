axis 是沿着第I个下标开始变化  

print arr[1, ...]               # 等价于 arr[1, :, :]  
print arr[..., 1]               # 等价于 arr[:, :, 1]  

ndarray = numpy.pad(array, pad_width, mode, **kwargs)  
https://www.cnblogs.com/hezhiyao/p/8177541.html

array为要填补的数组
pad_width是在各维度的各个方向上想要填补的长度,如（（1，2），（2，2）），表示在第一个维度上水平方向上padding=1,垂直方向上padding=2,在第二个维度上水平方向上padding=2,垂直方向上padding=2。如果直接输入一个整数，则说明各个维度和各个方向所填补的长度都一样。
 mode为填补类型，即怎样去填补，有“constant”，“edge”等模式，如果为constant模式，就得指定填补的值，如果不指定，则默认填充0。 
剩下的都是一些可选参数，具体可查看 
https://docs.scipy.org/doc/numpy/reference/generated/numpy.pad.html
ndarray为填充好的返回值。

