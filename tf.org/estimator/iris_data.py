import pandas as pd
import tensorflow as tf 

CSV_COLUMN_NAMES = ['SepalLength', 'SepalWidth',
                    'PetalLength', 'PetalWidth', 'Species']
SPECIES = ['Setosa', 'Versicolor', 'Virginica']

def load_data(y_name='Species'):
    train = pd.read_csv('./iris_training.csv', header=0, names=CSV_COLUMN_NAMES)
    train_x, train_y = train, train.pop(y_name)
    
    test = pd.read_csv('./iris_test.csv', names=CSV_COLUMN_NAMES, header=0)
    test_x, test_y = test, test.pop(y_name)

    return (train_x, train_y), (test_x, test_y)


def train_input_fn(features, labels, batch_size):
    ds = tf.data.Dataset.from_tensor_slices((dict(features), labels))
    ds = ds.shuffle(1000).repeat().batch(batch_size)
    return ds


def evaluate_input_fn(features, labels, batch_size):
    ds = tf.data.Dataset.from_tensor_slices((dict(features), labels))
    ds = ds.batch(batch_size)
    return ds


if __name__ == '__main__':
    (train_x, train_y), (test_x, test_y) = load_data()
    print(train_x.keys())

