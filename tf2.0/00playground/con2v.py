#coding: utf-8
import tensorflow as tf 

# [(n + 2p -f)/s + 1] x [(n + 2p -f)/s + 1]

# x_train_tensor=Input(shape=(28, 28, 1), name='x_train')
x = tf.ones((1,4,4,3), dtype='float64')
print(x.shape) #(1, 4, 4, 3)

layer = tf.keras.layers.Conv2D(filters=3, kernel_size=(3,3), dtype='float64',data_format='channels_last')

y = layer(x)
print(y.shape) #(1, 2, 2, 3)
f = tf.keras.layers.Flatten()(y)
print(f.shape) #(1, 12)
# print(layer)

# filters: 整数，输出空间的维度 （即卷积中滤波器的数量）。
# kernel_size: 一个整数，或者 2 个整数表示的元组或列表， 指明 2D 卷积窗口的宽度和高度。 可以是一个整数，为所有空间维度指定相同的值。
# strides: 一个整数，或者 2 个整数表示的元组或列表， 指明卷积沿宽度和高度方向的步长。 可以是一个整数，为所有空间维度指定相同的值。 指定任何 stride 值 != 1 与指定 dilation_rate 值 != 1 两者不兼容。
# padding: "valid" 或 "same" (大小写敏感)。   valid padding就是不padding，而same padding就是指padding完尺寸与原来相同
# 图像识别一般来说都要padding，尤其是在图片边缘的特征重要的情况下。padding多少取决于我们需要的输出是多少
# data_format: 字符串， channels_last (默认) 或 channels_first 之一，表示输入中维度的顺序。 channels_last 对应输入尺寸为 (batch, height, width, channels)， channels_first 对应输入尺寸为 (batch, channels, height, width)。 它默认为从 Keras 配置文件 ~/.keras/keras.json 中 找到的 image_data_format 值。 如果你从未设置它，将使用 channels_last。
# dilation_rate: 一个整数或 2 个整数的元组或列表， 指定膨胀卷积的膨胀率。 可以是一个整数，为所有空间维度指定相同的值。 当前，指定任何 dilation_rate 值 != 1 与 指定 stride 值 != 1 两者不兼容。
# activation: 要使用的激活函数 。 如果你不指定，则不使用激活函数 (即线性激活： a(x) = x)。
# use_bias: 布尔值，该层是否使用偏置向量。
# kernel_initializer: kernel 权值矩阵的初始化器 。
# bias_initializer: 偏置向量的初始化器 。
# kernel_regularizer: 运用到 kernel 权值矩阵的正则化函数 。
# bias_regularizer: 运用到偏置向量的正则化函数 。
# activity_regularizer: 运用到层输出（它的激活值）的正则化函数 。
# kernel_constraint: 运用到 kernel 权值矩阵的约束函数 。
# bias_constraint: 运用到偏置向量的约束函数

# 输入尺寸
# 如果 data_format='channels_first'， 输入 4D 张量，尺寸为 (samples, channels, rows, cols)。
# 如果 data_format='channels_last'， 输入 4D 张量，尺寸为 (samples, rows, cols, channels)。

# 输出尺寸
# 如果 data_format='channels_first'， 输出 4D 张量，尺寸为 (samples, filters, new_rows, new_cols)。
# 如果 data_format='channels_last'， 输出 4D 张量，尺寸为 (samples, new_rows, new_cols, filters)。