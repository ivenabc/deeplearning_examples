import tensorflow as tf 

numeric_feature_column = tf.feature_column.numeric_column(key="SepalLength")
vector_feature_column = tf.feature_column.numeric_column(key="Bowling",
                                                         shape=10)
matrix_feature_column = tf.feature_column.numeric_column(key='MyMatrix', shape=[10,5])

year_feature_column = tf.feature_column.numeric_column('Year')

bucketized_feature_column = tf.feature_column.bucketized_column(
    source_column = year_feature_column,
    boundaries = [1960, 1980, 2000])

identity_feature_column = tf.feature_column.categorical_column_with_identity(
    key='my_feature_b',
    num_buckets=4)

def input_fn():
    return ({ 'my_feature_a':[7, 9, 5, 2], 'my_feature_b':[3, 1, 2, 2] },
            [Label_values])


if __name__ == "__main__":
    
    