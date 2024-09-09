"""
Author: Abdalraheem A. Ijjeh
Affiliation: Postdoc researcher / GRVC Robotics Laboratory, University of Seville, 41092 Seville, Spain

Description: This script reads a TensorFlow TFRecord file containing images and annotations, decodes the images,
and extracts bounding boxes and class labels. It then uses OpenCV to draw bounding boxes and labels on the images and
displays them.

The script performs the following tasks:
- Reads and parses a TFRecord file.
- Decodes images and retrieves bounding box coordinates and labels.
- Draws bounding boxes and labels on the images using OpenCV.
- Displays the annotated images.

Key Features:
- TFRecord Parsing: Parses a TFRecord file to extract image data and annotations.
- Image Decoding: Decodes JPEG images and retrieves bounding box coordinates.
- Bounding Box Drawing: Draws bounding boxes and labels on images using OpenCV.
- Visualization: Displays images with bounding boxes and labels.

Requirements:
- TensorFlow: For reading and parsing TFRecord files.
- OpenCV: For drawing bounding boxes and displaying images.
- NumPy: For numerical operations (implicitly used via TensorFlow).

Usage:
1. TFRecord File Path: Set `tfrecord_file` to the path of the TFRecord file to be processed.
2. Parsing and Decoding: The script reads and parses the TFRecord file, then decodes images and extracts annotations.
3. Drawing and Displaying: Uses OpenCV to draw bounding boxes and labels on the images and displays them.

Example Command:
```python
python VisTFRecord.py
```

Notes:
- Ensure the TFRecord file specified in `tfrecord_file` exists and contains the required features
(`image_raw`, `height`, `width`, `x_mins`, `x_maxes`, `y_mins`, `y_maxes`, `classes_id`, `classes_name`).
- The script currently draws bounding boxes and labels for a class named "person" (or other labels from the dataset).
Adjust the class name or labels as needed.
- To process and display a large number of images, modify the `decoded_images.take(1000)` line to suit your needs.

"""
import tensorflow as tf
import cv2

# Specify the path to your TFRecord file
tfrecord_file = 'dataset/TotalThermalImages.tfrecord'

# Define the feature description dictionary
feature_description = {
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


# Function to parse the example
def _parse_function(proto):
    return tf.io.parse_single_example(proto, feature_description)


# Create a TFRecordDataset
raw_dataset = tf.data.TFRecordDataset(tfrecord_file)

# Parse the dataset
parsed_dataset = raw_dataset.map(_parse_function)
# parsed_dataset = parsed_dataset.shuffle(buffer_size=512)


# Function to decode the image and extract bounding boxes and labels
def _decode_image_and_labels(parsed_record):
    image = tf.image.decode_jpeg(parsed_record['image_raw'], channels=3)
    # image = tf.image.decode_bmp(parsed_record['image_raw'], channels=3)
    height = tf.cast(parsed_record['height'], tf.float32)
    width = tf.cast(parsed_record['width'], tf.float32)

    # Retrieve normalized bounding box coordinates
    xmins = tf.sparse.to_dense(parsed_record['x_mins'])
    xmaxs = tf.sparse.to_dense(parsed_record['x_maxes'])
    ymins = tf.sparse.to_dense(parsed_record['y_mins'])
    ymaxs = tf.sparse.to_dense(parsed_record['y_maxes'])

    # Convert normalized coordinates to pixel values
    xmin = xmins  # * width
    xmax = xmaxs  # * width
    ymin = ymins  # * height
    ymax = ymaxs  # * height

    classes_ID = tf.sparse.to_dense(parsed_record['classes_id'])
    classes_Name = tf.sparse.to_dense(parsed_record['classes_name'])

    return image, xmin, xmax, ymin, ymax, classes_ID, height, width


# Decode the images and extract bounding boxes and labels
decoded_images = parsed_dataset.map(_decode_image_and_labels)


# Function to draw bounding boxes and labels using OpenCV
def draw_bboxes_with_labels(image, xmin, xmax, ymin, ymax, labels, h, w):
    # Convert the image to a format suitable for OpenCV (BGR)
    image = cv2.cvtColor(image.numpy(), cv2.COLOR_RGB2BGR)
    labels = tf.strings.as_string(labels)
    for i in range(labels.shape[0]):
        xmin_i = int((2*xmin[i]-xmax[i]) * w)
        xmax_i = int(xmax[i] * w)
        ymin_i = int((2*ymin[i]-ymax[i]) * h)
        ymax_i = int(ymax[i] * h)
        label = str(labels[i].numpy().decode('utf-8'))

        cv2.rectangle(image, (xmin_i, ymin_i), (xmax_i, ymax_i), (0, 255, 0), 2)

        # Put label text
        cv2.putText(image, 'person',
                    (xmin_i, ymin_i - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 255, 0),
                    2)

    # Display the image
    cv2.imshow('Image with BBoxes', image)
    cv2.waitKey()
    cv2.destroyAllWindows()


# Iterate through the dataset and draw bounding boxes and labels on the images
for image, xmin, xmax, ymin, ymax, labels, h, w in decoded_images.take(
        1000):  # Take the first 5 images for example
    draw_bboxes_with_labels(image, xmin, xmax, ymin, ymax, labels.numpy(), h, w)
