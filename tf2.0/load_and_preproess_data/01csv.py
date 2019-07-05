import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

TRAIN_DATA_URL = "https://storage.googleapis.com/tf-datasets/titanic/train.csv"
TEST_DATA_URL = "https://storage.googleapis.com/tf-datasets/titanic/eval.csv"

train_file_path = tf.keras.utils.get_file(
    "train.csv", TRAIN_DATA_URL, cache_dir='./data')
test_file_path = tf.keras.utils.get_file(
    "eval.csv", TEST_DATA_URL,  cache_dir='./data')

with open(train_file_path, 'r') as f:
    names_row = f.readline()

CSV_COLUMNS = names_row.rstrip('\n').split(',')
print(CSV_COLUMNS)

LABELS = [0, 1]
LABEL_COLUMN = 'survived'

FEATURE_COLUMNS = [column for column in CSV_COLUMNS if column != LABEL_COLUMN]


def get_dataset(file_path):
    dataset = tf.data.experimental.make_csv_dataset(
        file_path,
        batch_size=12,  # Artificially small to make examples easier to show.
        label_name=LABEL_COLUMN,
        na_value="?",
        num_epochs=1,
        ignore_errors=True)
    return dataset


CATEGORIES = {
    'sex': ['male', 'female'],
    'class': ['First', 'Second', 'Third'],
    'deck': ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J'],
    'embark_town': ['Cherbourg', 'Southhampton', 'Queenstown'],
    'alone': ['y', 'n']
}

MEANS = {
    'age': 29.631308,
    'n_siblings_spouses': 0.545455,
    'parch': 0.379585,
    'fare': 34.385399
}


def process_categorical_data(data, categories):

    # Remove leading ' '.
    data = tf.strings.regex_replace(data, '^ ', '')
    # Remove trailing '.'.
    data = tf.strings.regex_replace(data, r'\.$', '')

    # ONE HOT ENCODE
    # Reshape data from 1d (a list) to a 2d (a list of one-element lists)
    data = tf.reshape(data, [-1, 1])
    # For each element, create a new list of boolean values the length of categories,
    # where the truth value is element == category label
    data = tf.equal(categories, data)
    # Cast booleans to floats.
    data = tf.cast(data, tf.float32)

    # The entire encoding can fit on one line:
    # data = tf.cast(tf.equal(categories, tf.reshape(data, [-1, 1])), tf.float32)
    return data


def process_continuous_data(data, mean):
    data = tf.cast(data, tf.float32) * 1 / (2*mean)
    return tf.reshape(data, [-1, 1])


def preprocess(features, labels):
    for feature in CATEGORIES.keys():
        features[feature] = process_categorical_data(
            features[feature], CATEGORIES[feature])

    for feature in MEANS.keys():
        features[feature] = process_continuous_data(
            features[feature], MEANS[feature])

    # features = tf.concat([features[column] for column in FEATURE_COLUMNS])
    print('preprocess:\n', [features[column] for column in FEATURE_COLUMNS])
    features = tf.concat([features[column] for column in FEATURE_COLUMNS], 1)
    print("features", features)
    return features, labels


raw_train_data = get_dataset(train_file_path)
raw_test_data = get_dataset(test_file_path)


train_data = raw_train_data.map(preprocess).shuffle(500)
test_data = raw_test_data.map(preprocess)


def get_model(input_dim, hidden_units=[100]):
    inputs = tf.keras.Input(shape=(input_dim,))
    x = inputs

    for units in hidden_units:
        x = tf.keras.layers.Dense(units, activation='relu')(x)

    outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)
    model = tf.keras.Model(inputs, outputs)

    return model


input_shape, output_shape = train_data.output_shapes

input_dimension = input_shape.dims[1]  # [0] is the batch size

print(input_dimension)
# print('traindata[0]', train_data[0])
model = get_model(input_dimension)
model.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)
model.fit(train_data, epochs=20)

test_loss, test_accuracy = model.evaluate(test_data)

print('\n\nTest Loss {}, Test Accuracy {}'.format(test_loss, test_accuracy))


# predictions = model.predict(test_data)

# for prediction, survived in zip(predictions[:10], list(test_data)[0][1][:10]):
#     print("Predicted survival: {:.2%}".format(prediction[0]),
#           " | Actual outcome: ",
#           ("SURVIVED" if bool(survived) else "DIED"))

# itr = iter(train_data)
# features, labels = next(itr)
# print("labels====>", labels)
# print("features====>", features)
# features, labels = next(itr)
# print("labels====>", labels)
# features, labels = next(itr)
# print("labels====>", labels)
# features, labels = next(itr)
# print("labels====>", labels)


# itr = iter(raw_train_data)
# examples, labels = next(itr)
# print("EXAMPLES: \n", examples, "\n")
# print("LABELS: \n", labels)

# class_tensor = examples['class']
# class_categories = CATEGORIES['class']
# print('class_tensor', class_tensor)
# print('class_categories', class_categories)

# process_class =  process_categorical_data(class_tensor, class_categories)
# print('process_class', process_class)


predictions = model.predict(test_data)

# Show some results
for prediction, survived in zip(predictions[:10], list(test_data)[0][1][:10]):
  print("Predicted survival: {:.2%}".format(prediction[0]),
        " | Actual outcome: ",
        ("SURVIVED" if bool(survived) else "DIED"))