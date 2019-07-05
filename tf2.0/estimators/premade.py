import tensorflow as tf 


CSV_COLUMN_NAMES = ['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth', 'Species']
SPECIES = ['Setosa', 'Versicolor', 'Virginica']

TRAIN_DATA_URL = 

train_path =  tf.keras.utils.get_file(
    "train.csv", TRAIN_DATA_URL, cache_dir='./data')

test_path = tf.keras.utils.get_file(
    "eval.csv", TEST_DATA_URL,  cache_dir='./data')