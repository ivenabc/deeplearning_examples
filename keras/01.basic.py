from tensorflow import keras


# 使用keras 开发流程
# 1) 定义训练数据：输入张量和目标张量。
# (2) 定义层组成的网络（或模型），将输入映射到目标。
# (3) 配置学习过程：选择损失函数、优化器和需要监控的指标。
# (4) 调用模型的 fit 方法在训练数据上进行迭代。

# Sequential 仅用于层的线性堆叠，这是目前最常见的网络架构
model = keras.models.Sequential()

model.add(keras.layers.Dense(32, activation='relu', input_shape=(784, )))
model.add(keras.layers.Dense(10, activation='softmax'))


input_tensor = keras.layers.Input(shape=(784,))
x = keras.layers.Dense(32, activation='relu')(input_tensor)
output_tensor = keras.layers.Dense(10, activation='softmax')(x)

model = keras.models.Model(inputs=input_tensor, outputs=output_tensor)