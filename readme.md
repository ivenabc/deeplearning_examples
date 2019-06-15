
### 指定国内链接安装包
```
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple pillow
```

### 设置代理安装包
```
export http_proxy=socks5://127.0.0.1:1080
export https_proxy=socks5://127.0.0.1:1080
export ftp_proxy=socks5://127.0.0.1:1080
```

### numpy基本操作
```
import numpy as np
 
b = np.array([[1,2,3],[4,5,6],[7,8,9],[10,11,12]],dtype=int)
 
c = b[0,1]  #1行 第二个单元元素
# 输出： 2
d = b[:,1]  #所有行 第二个单元元素
# 输出： [ 2  5  8 11]
e = b[1,:]  #2行 所有单元元素
# 输出： [4 5 6]
f = b[1,1:]  #2行 第2个单元开始以后所有元素
# 输出： [5 6]
g = b[1,:2]  #2行 第1个单元开始到索引为2以前的所有元素

传入顺序索引数组
h = b[[1,2,4,5,6]]

```

### numpy 版本
```
pip install numpy==1.16.2
```
《深度学习入门：基于Python的理论与实现》