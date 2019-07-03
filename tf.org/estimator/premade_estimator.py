from iris_data import load_data, train_input_fn, evaluate_input_fn
import tensorflow as tf

if __name__ == "__main__":
    (train_x, train_y), (test_x, test_y) = load_data()
    my_feature_columns = []
    for key in train_x.keys():
        my_feature_columns.append(tf.feature_column.numeric_column(key=key))

    classifier = tf.estimator.DNNClassifier(
        feature_columns=my_feature_columns,
        hidden_units=[10, 10],
        n_classes=3,
    )

    print('train start')
    classifier.train(
        input_fn=lambda: train_input_fn(train_x, train_y,30),
        steps=100)
    print('train end')
    eval_result = classifier.evaluate(
        input_fn=lambda: evaluate_input_fn(test_x, test_y,30),
    )
    print('\nTest set accuracy: {accuracy:0.3f}\n'.format(**eval_result))