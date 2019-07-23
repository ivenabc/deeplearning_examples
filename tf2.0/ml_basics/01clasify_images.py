from PIL import Image
import numpy as np 
import tensorflow as tf 


def img_show(img):
    
    pil_img = Image.fromarray(np.uint8(img))
    pil_img.show()
    # pil_img.save('./tmp.png')


fashion_mnist = tf.keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# img_show(train_images[0])

train_images = train_images / 255.0
test_images = test_images / 255.0

model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28,28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


model.fit(train_images, train_labels, epochs=10)

test_loss, test_acc = model.evaluate(test_images, test_labels)
print('\n Test accuracy:', test_acc)

print('\n Test loss:', test_loss)