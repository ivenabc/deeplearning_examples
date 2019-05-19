import tensorflow as tf 

# print(tf.random_uniform([4, 10]))
# dataset1 = tf.data.Dataset.from_tensor_slices(tf.random.normal([4, 10]))

# print('dataset types:', dataset1.output_types)
# print('dataset shapes:', dataset1.output_shapes)


dataset = tf.data.Dataset.range(100)

iterator = dataset.make_one_shot_iterator()

next_element = iterator.get_next()

with tf.Session() as sess:
    for i in range(100):
        value = sess.run(next_element)
        print(value)
    # for i in range(100):
    #     value = sess.run(next_element)
    #     print(value)