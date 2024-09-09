"""
Author: Abdalraheem A. Ijjeh
Affiliation: Postdoc researcher / GRVC Robotics Laboratory, University of Seville, 41092 Seville, Spain

Description:
    This script performs anchor box calculation using K-Means clustering on bounding box dimensions
    extracted from a custom dataset in TFRecord format. The extracted bounding box dimensions are used
    to compute optimized anchor sizes that can improve object detection performance.

    The script includes the following steps:
    - Parsing of the TFRecord files to extract bounding box dimensions.
    - Calculation of anchor boxes using K-Means clustering.

    Key Features:
    - Custom dataset loading and TFRecord parsing.
    - Bounding box extraction (height, width).
    - K-Means clustering to compute anchor boxes.

Usage:
    1. Update `tfrecord_paths` to point to your training and validation TFRecord files.
    2. Set the number of anchors (`num_anchors`) to calculate.
    3. Run the script to compute anchor boxes, which will be printed out for verification.

Requirements:
    - TensorFlow
    - Scikit-learn
"""
import tensorflow as tf
import numpy as np
from sklearn.cluster import KMeans


# Function to parse TFRecord
def parse_tfrecord_fn(record):
    features = {
        'filename': tf.io.FixedLenFeature([], tf.string),
        'height': tf.io.FixedLenFeature([], tf.int64),
        'width': tf.io.FixedLenFeature([], tf.int64),
        'classes_id': tf.io.VarLenFeature(tf.int64),
        'classes_name': tf.io.VarLenFeature(tf.string),
        'x_mins': tf.io.VarLenFeature(tf.float32),
        'y_mins': tf.io.VarLenFeature(tf.float32),
        'x_maxes': tf.io.VarLenFeature(tf.float32),
        'y_maxes': tf.io.VarLenFeature(tf.float32),
        'image_raw': tf.io.FixedLenFeature([], tf.string),
    }
    parsed_features = tf.io.parse_single_example(record, features)
    return parsed_features


# Function to extract bounding box dimensions from parsed TFRecord
def extract_bounding_boxes(parsed_features):
    height = tf.cast(parsed_features['height'], tf.float32)
    width = tf.cast(parsed_features['width'], tf.float32)

    # Retrieve normalized bounding box coordinates
    xmins = tf.sparse.to_dense(parsed_features['x_mins'])
    xmaxs = tf.sparse.to_dense(parsed_features['x_maxes'])
    ymins = tf.sparse.to_dense(parsed_features['y_mins'])
    ymaxs = tf.sparse.to_dense(parsed_features['y_maxes'])

    bboxes = []
    for xmin, xmax, ymin, ymax in zip(xmins, xmaxs, ymins, ymaxs):
        box_width = (xmax - xmin) * width
        box_height = (ymax - ymin) * height
        bboxes.append([box_width, box_height])

    return bboxes


# Load dataset and extract bounding boxes
def load_dataset(tfrecord_paths_):
    dataset_ = []
    raw_dataset = tf.data.TFRecordDataset(tfrecord_paths_)
    for raw_record in raw_dataset:
        parsed_features = parse_tfrecord_fn(raw_record)
        bboxes = extract_bounding_boxes(parsed_features)
        dataset_.extend(bboxes)

    return np.array(dataset_)


# Paths to your TFRecord files
tfrecord_paths = ["dataset/train_single_class_person.tfrecord", "dataset/val_single_class_person.tfrecord"]
dataset = load_dataset(tfrecord_paths)

# Save the extracted dataset for verification
# np.save('bbox_dimensions.npy', dataset)

# Number of anchors you want to calculate
num_anchors = 9


# K-Means clustering to find anchors
def calculate_anchors(dataset, num_anchors):
    kmeans = KMeans(n_clusters=num_anchors, random_state=0).fit(dataset)
    return kmeans.cluster_centers_


anchors = calculate_anchors(dataset, num_anchors)
print("Calculated anchors (width, height):")
print(anchors)


