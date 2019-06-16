import tensorflow as tf 

dataset = tf.data.Dataset.range(10)
for i, elem in enumerate(dataset):
    print(i, elem)


it = iter(dataset)
print(next(it))

dataset = tf.data.Dataset.from_tensor_slices([8,3,0,8,2,1])
print(dataset.reduce(0, lambda state, value: state + value))

dataset2 = tf.data.Dataset.from_tensor_slices((
    tf.random.uniform([4], maxval=100,dtype=tf.int32), 
    tf.random.uniform([4,100], maxval=100, dtype=tf.int32)
    ))
# it2 = iter(dataset2)
# print("=================================")
# print(next(it2))
# print("=================================")
# print(next(it2))
# print("=================================")
# print(next(it2))
# print("=================================")

dataset2V1 = dataset2.flat_map(lambda x,y: tf.data.Dataset.from_tensor_slices([x]))

it2v1 = iter(dataset2V1)
print("it2v1==================================")
print(next(it2v1))


print("it3==================================")
dataset3 = tf.data.Dataset.from_tensor_slices(tf.random.uniform([4,10]))
it3 = iter(dataset3)
print(next(it3))