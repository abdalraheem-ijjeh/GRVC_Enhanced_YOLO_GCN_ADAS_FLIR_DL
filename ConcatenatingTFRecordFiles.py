"""
Author: Abdalraheem A. Ijjeh
Affiliation: Postdoc researcher / GRVC Robotics Laboratory, University of Seville, 41092 Seville, Spain

Description:
    This script processes TensorFlow TFRecord files to handle bounding box data. It includes functions to
    parse TFRecord files, create datasets, serialize data, and concatenate TFRecord files. It also
    provides a utility to count the number of samples in a TFRecord file. This script is useful for
    managing and processing data in TFRecord format for machine learning tasks.

    The script includes the following steps:
    - Parsing TFRecord examples to extract features.
    - Creating a dataset from a list of TFRecord files.
    - Serializing parsed features into TFRecord format.
    - Saving concatenated TFRecord files.
    - Counting the number of samples in a TFRecord file.

    Key Features:
    - Parsing of TFRecord files using TensorFlow's `tf.data` API.
    - Serialization of data into TFRecord format.
    - Concatenation of multiple TFRecord files into a single file.
    - Counting the number of records in a TFRecord file.

Usage:
    1. Update `tfrecord_files` to include paths to the TFRecord files you want to process.
    2. Specify the `output_file` where the concatenated TFRecord file will be saved.
    3. Run the script to concatenate the TFRecord files and save the result to `output_file`.

Requirements:
    - TensorFlow
"""


import tensorflow as tf


def _parse_function(proto):
    keys_to_features = {
        'filename': tf.io.FixedLenFeature([], tf.string),
        'height': tf.io.FixedLenFeature([], tf.int64),
        'width': tf.io.FixedLenFeature([], tf.int64),
        'image_raw': tf.io.FixedLenFeature([], tf.string),
        'image_id': tf.io.FixedLenFeature([], tf.int64),
        'classes_id': tf.io.VarLenFeature(tf.int64),
        'classes_name': tf.io.VarLenFeature(tf.string),
        'x_mins': tf.io.VarLenFeature(tf.float32),
        'y_mins': tf.io.VarLenFeature(tf.float32),
        'x_maxes': tf.io.VarLenFeature(tf.float32),
        'y_maxes': tf.io.VarLenFeature(tf.float32),
    }

    parsed_features = tf.io.parse_single_example(proto, keys_to_features)
    return parsed_features


def create_dataset(tfrecord_files):
    raw_dataset = tf.data.TFRecordDataset(tfrecord_files)
    parsed_dataset = raw_dataset.map(_parse_function)
    # shuffled_dataset = parsed_dataset.shuffle(buffer_size=1000)  # Adjust buffer size as needed
    return parsed_dataset


def _serialize_example(parsed_features):
    features = {
        'filename': tf.train.Feature(bytes_list=tf.train.BytesList(value=[parsed_features['filename'].numpy()])),
        'height': tf.train.Feature(int64_list=tf.train.Int64List(value=[parsed_features['height'].numpy()])),
        'width': tf.train.Feature(int64_list=tf.train.Int64List(value=[parsed_features['width'].numpy()])),
        'image_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[parsed_features['image_raw'].numpy()])),
        'image_id': tf.train.Feature(int64_list=tf.train.Int64List(value=[parsed_features['image_id'].numpy()])),
        'classes_id': tf.train.Feature(
            int64_list=tf.train.Int64List(value=parsed_features['classes_id'].values.numpy())),
        'classes_name': tf.train.Feature(
            bytes_list=tf.train.BytesList(value=parsed_features['classes_name'].values.numpy())),
        'x_mins': tf.train.Feature(float_list=tf.train.FloatList(value=parsed_features['x_mins'].values.numpy())),
        'y_mins': tf.train.Feature(float_list=tf.train.FloatList(value=parsed_features['y_mins'].values.numpy())),
        'x_maxes': tf.train.Feature(float_list=tf.train.FloatList(value=parsed_features['x_maxes'].values.numpy())),
        'y_maxes': tf.train.Feature(float_list=tf.train.FloatList(value=parsed_features['y_maxes'].values.numpy())),
    }

    example_proto = tf.train.Example(features=tf.train.Features(feature=features))
    return example_proto.SerializeToString()


def save_concatenated_tfrecord(dataset, output_file):
    writer = tf.io.TFRecordWriter(output_file)
    for parsed_features in dataset:
        serialized_example = _serialize_example(parsed_features)
        writer.write(serialized_example)
    writer.close()


def count_samples_in_tfrecord(tfrecord_files):
    raw_dataset = tf.data.TFRecordDataset(tfrecord_files)
    count = raw_dataset.reduce(0, lambda x, _: x + 1).numpy()
    return count


print(count_samples_in_tfrecord('dataset/val_single_class_person.tfrecord'))
exit()

tfrecord_files = ['dataset/images_thermal_val_single_class_person_center.tfrecord',
                  'dataset/images_rgb_val_single_class_person_center.tfrecord']
dataset = create_dataset(tfrecord_files)
output_file = 'dataset/val_single_class_person_center.tfrecord'
save_concatenated_tfrecord(dataset, output_file)

print(f"Concatenated TFRecord file saved to: {output_file}")
