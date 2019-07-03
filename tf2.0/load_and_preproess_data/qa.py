import tensorflow as tf 


# tf.reshape(tensor, shape, name=None) 
# 函数的作用是将tensor变换为参数shape的形式。 
# 其中shape为一个列表形式，特殊的一点是列表中可以存在-1。-1代表的含义是不用我们自己指定这一维的大小，函数会自动计算，但列表中只能存在一个-1

# tf.math.equal(x,y,name=None) 比较两个矩阵是否相等 ，同时支持广播模式。
# 如果两个维度都相同 则 一对一比较 返回 x
# 如果两个维度不相同， 拿y的每一个元素 与 x进行广播 得到的结果 是 y的 一个最小维度的单元 

t = tf.Variable([1, 2, 3, 4],  tf.int32)

print('tf.reshape(t, [-1, 1]):\n',  tf.reshape(t, [-1, 1]))

c = ['First', 'Second', 'Third']

t = [ 'Second', 'Third', 'First', 'First', 'Second']
t = tf.reshape(t, [-1, 1])
r = tf.equal(c, t)

print('tf.equal(c, t):', r)
print('tf.cast(r)', tf.cast(r, tf.int32))
# print('tf.reshape(t, []):\n',  tf.reshape(t, []))  error 