import tensorflow as tf 
import numpy as np 
import tensorflow_hub as hub
import tensorflow_datasets as tfds


print("Version: ", tf.__version__)
print("Eager mode: ", tf.executing_eagerly())
print("Hub version: ", hub.__version__)
print("GPU is", "available" if tf.test.is_gpu_available() else "NOT AVAILABLE")