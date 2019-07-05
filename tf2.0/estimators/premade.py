import tensorflow as tf
import tensorflow.compat.v2.feature_column as fc

CSV_COLUMN_NAMES = ['SepalLength', 'SepalWidth',
    'PetalLength', 'PetalWidth', 'Species']
SPECIES = ['Setosa', 'Versicolor', 'Virginica']

TRAIN_DATA_URL = 'https://storage.googleapis.com/download.tensorflow.org/data/iris_training.csv'
TEST_DATA_URL = 'https://storage.googleapis.com/download.tensorflow.org/data/iris_test.csv'

train_path = tf.keras.utils.get_file(
    "train.csv", TRAIN_DATA_URL, cache_dir='./data')

test_path = tf.keras.utils.get_file(
    "eval.csv", TEST_DATA_URL,  cache_dir='./data')

BATCH_SIZE = 30

def get_dataset(file_path):
    dataset = tf.data.experimental.make_csv_dataset(
        file_path,
        column_names=CSV_COLUMN_NAMES,
        batch_size=BATCH_SIZE,
        label_name='Species',
        # na_value="?",
        num_epochs=1,
        ignore_errors=False)
    return dataset


train_raw_data = get_dataset(train_path)
train_iter = iter(train_raw_data)
train_data, labels = next(train_iter)
print(train_data)
# print('====================================\n')
# print(labels)

def input_fn(features, lables):
    # features, labels = next(preds)

    # pre_data = [featuers[key] for key in featuers.keys()]
    # print('pre_Data', pre_data)
    # ds = tf.concat([pre_data], 1)
    # print('featuers:', featuers)
    # ds = tf.data.Dataset.from_tensor_slices((dict(featuers)))
    # print('ds:', ds)
    # print('======================================================')
    # print(next(iter(ds)))
    # return ds, labels
    dataset = tf.data.Dataset.from_tensor_slices((features, labels))

    dataset = dataset.shuffle(1000).repeat()
    dataset = tf.cast(dataset, tf.float32)
    # Shuffle and repeat if you are in training mode.
    return dataset.batch(BATCH_SIZE)

# def get_train_data():
#     return train_raw_data

# # print('ds:', input_fn(train_data, labels))
columns = [tf.feature_column.numeric_column(key=key) for key in  ['SepalLength', 'SepalWidth',
    'PetalLength', 'PetalWidth']]
print(columns)
# # Build a DNN with 2 hidden layers with 30 and 10 hidden nodes each.
classifier = tf.estimator.DNNClassifier(
    feature_columns=columns,
    # Two hidden layers of 10 nodes each.
    hidden_units=[30, 10],
    # The model must choose between 3 classes.
    n_classes=3)

classifier.train(
    input_fn=lambda: input_fn(train_data, labels),
    steps=100)