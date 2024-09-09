"""
Author: Abdalraheem A. Ijjeh
Affiliation: Postdoc researcher / GRVC Robotics Laboratory, University of Seville, 41092 Seville, Spain

Description: This script converts image and annotation data into TensorFlow TFRecord format, which is commonly used
for training machine learning models. It processes images and annotations to create TFRecord files with features such
as image dimensions, raw image data, and bounding box annotations.

The script performs the following tasks:
- Read images and annotations from a specified directory.
- Convert image and annotation data into TensorFlow `tf.train.Example` format.
- Write the `tf.train.Example` objects into TFRecord files.

Key Features: - Image and Annotation Conversion: Converts images and their annotations into TensorFlow
`tf.train.Example` format suitable for training models. - Bounding Box Normalization: Normalizes bounding box
coordinates to relative values between 0 and 1 based on image dimensions. - Single Class Handling: Focuses on
annotations for a specific class (e.g., person) for conversion to TFRecord format.

Requirements:
- TensorFlow: For creating and writing TFRecord files.
- JSON: For reading annotation data.
- OS: For file and directory operations.

Usage:
1. Data Directory: Set `data_dir` to the path containing your image and annotation data.
2. Output Directory: Specify `output_dir` where the TFRecord files will be saved.
3. Image Types: Define which image types (e.g., 'images_thermal_val', 'images_thermal_train') to process.
4. Create TFRecord: Calls `create_tfrecord()` to convert images and annotations to TFRecord format.

Example Command:
```python
python TFRecordCreationCode.py
```

Notes:
- Ensure that the `data_dir` contains the images and a JSON file with annotations in COCO format.
- The script assumes that the annotations file is named 'coco.json' and is located in the same directory as the images.
- Adjust the `person_class` variable if you need to handle a different class or multiple classes.

"""
import tensorflow as tf
import os
import json


def image_example(image_path, height, width, image_id, annotation, class_names, counter):
    with open(image_path, 'rb') as img_file:
        image_data = img_file.read()

    feature = {
        'filename': tf.train.Feature(
            bytes_list=tf.train.BytesList(value=[os.path.basename(image_path).encode('utf-8')])),
        'height': tf.train.Feature(int64_list=tf.train.Int64List(value=[height])),
        'width': tf.train.Feature(int64_list=tf.train.Int64List(value=[width])),
        'image_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_data])),
        'image_id': tf.train.Feature(int64_list=tf.train.Int64List(value=[image_id])),  # Add this line

    }

    if annotation:
        xmins, ymins, xmaxs, ymaxs, class_ids, class_names_list = [], [], [], [], [], []

        for bbox_dict in annotation:
            bbox = bbox_dict['bbox']
            category_id = bbox_dict['category_id']
            class_name = class_names[category_id]

            # print(category_id)
            if category_id == 1:
                counter += 1
                class_ids.append(category_id)
                class_names_list.append(class_name.encode('utf-8'))

                x_min = int(bbox[0])
                y_min = int(bbox[1])
                bbox_width = int(bbox[2])
                bbox_height = int(bbox[3])

                # x_center = x_min + (bbox_width / 2.0)
                # y_center = y_min + (bbox_height / 2.0)

                xmins.append(x_min / width)
                xmaxs.append((x_min + bbox_width) / width)
                ymins.append(y_min / height)
                ymaxs.append((y_min + bbox_height) / height)
            else:
                continue

        feature['classes_id'] = tf.train.Feature(int64_list=tf.train.Int64List(value=class_ids))
        feature['classes_name'] = tf.train.Feature(bytes_list=tf.train.BytesList(value=class_names_list))
        feature['x_mins'] = tf.train.Feature(float_list=tf.train.FloatList(value=xmins))
        feature['y_mins'] = tf.train.Feature(float_list=tf.train.FloatList(value=ymins))
        feature['x_maxes'] = tf.train.Feature(float_list=tf.train.FloatList(value=xmaxs))
        feature['y_maxes'] = tf.train.Feature(float_list=tf.train.FloatList(value=ymaxs))

    return tf.train.Example(features=tf.train.Features(feature=feature))


def create_tfrecord(data_dir_, split, output_file_, person_counter):
    writer = tf.io.TFRecordWriter(output_file_)
    image_dir = os.path.join(data_dir_, split)
    annotation_file = os.path.join(data_dir_, split, 'coco.json')

    with open(annotation_file, 'r') as f:
        data = json.load(f)

    images = data['images']
    categories = {category['id']: category['name'] for category in data['categories']}
    annotations = data['annotations']
    images_dict = {image['id']: {'file_name': image['file_name'], 'height': image['height'], 'width': image['width']}
                   for image in images}
    image_annotations = {}
    for annotation in annotations:
        image_id = annotation['image_id']
        category_id = annotation['category_id']
        bbox = annotation['bbox']

        if image_id in image_annotations:
            image_annotations[image_id].append({'category_id': category_id, 'bbox': bbox})
        else:
            image_annotations[image_id] = [{'category_id': category_id, 'bbox': bbox}]

    for image_id, annotations in image_annotations.items():
        image_info = images_dict[image_id]
        file_path = os.path.join(image_dir, image_info['file_name'])
        if isinstance(file_path, bytes):
            file_path = file_path.decode('utf-8')

        # Debugging output
        # print(f"Processing image_id: {image_id}, file_path: {file_path}")

        try:
            tf_example = image_example(file_path, image_info['height'], image_info['width'], image_id, annotations,
                                       categories, person_counter)
            writer.write(tf_example.SerializeToString())
        except KeyError as e:
            # print(f"KeyError: {e} for image_id: {image_id}, file_path: {file_path}")
            continue

    writer.close()


person_class = 0
if __name__ == '__main__':
    data_dir = '/home/abdalraheem/Documents/ADAS_DATASET_V2'
    # data_dir = '/home/abdalraheem/PycharmProjects/ADAS_FLIR_DL/ThermalDatasets'
    output_dir = 'dataset/'
    image_types = ['images_thermal_val', 'images_thermal_train']

    for t in image_types:
        person_class = 0
        # print(f'Processing {t}')
        output_file = os.path.join(output_dir, f'{t}_single_class_person.tfrecord')
        create_tfrecord(data_dir, t, output_file, person_class)
        print('num of bboxes: ', person_class)
