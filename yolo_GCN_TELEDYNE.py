"""
Author: Abdalraheem A. Ijjeh
Affiliation: Postdoc researcher / GRVC Robotics Laboratory, University of Seville, 41092 Seville, Spain

Description:
This script trains a custom and enhanced version of YOLOv3 model for object detection by using global convolutions.
It uses TensorFlow and Keras for model development and integrates Neptune for experiment tracking.

The script performs the following tasks:
- Defines a YOLO model with global convolution layers (separable convolution operations).
- Configures and preprocesses datasets for training and validation.
- Sets up training parameters and callbacks.
- Logs metrics and training progress using Neptune and TensorBoard.
- Save the trained model and checkpoints.

Key Features:
- Global Convolutions: Enhances feature extraction with global convolution layers.
- Dataset Handling: Loads and preprocesses datasets from TFRecord files.
- Custom Callbacks: Integrates Neptune for experiment tracking and logging.
- Model Checkpoints: Saves best model checkpoints and training progress.

Requirements:
- TensorFlow: For model building, training, and evaluation.
- Keras: For defining and using the YOLOv3 model.
- NumPy: For numerical operations.
- Absl-Py: For command-line flag management.
- Neptune: For experiment tracking.
- OS: For managing file paths and environment variables.

Usage:
1. Set Flags: Configure model parameters and paths using command-line flags.
2. Prepare Datasets: Ensure the training and validation datasets are in the specified TFRecord format.
3. Run Training: Execute the script to start training with the configured settings.

Example Command:
```bash
python script.py --size 416 --epochs 1000 --batch_size 4 --num_classes 1 --classes model_data/adas_flir_classes.txt --anchors model_data/yolov3_anchors.txt --train_dataset dataset/images_thermal_train_single_class_person.tfrecord --val_dataset dataset/images_thermal_val_single_class_person.tfrecord --output model_data/yolov3.keras
```

Notes:
- Ensure that Neptune API tokens are set in your environment for logging.
- The script assumes that the model weights are in Keras format (.h5) for loading and transfer learning.
- Adjust the number of classes, dataset paths, and other parameters as needed based on your specific use case.

"""

import math
import os

import numpy as np
from keras.layers import Conv2D
from keras.layers import Input
from keras.layers import BatchNormalization
from keras.layers import LeakyReLU
from keras.layers import UpSampling2D
from absl import app, flags, logging
from absl.flags import FLAGS
import tensorflow as tf
from keras.models import Model
from tensorflow.keras.layers import concatenate, add, Add
from tensorflow.keras.callbacks import Callback, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, TensorBoard
from utils.common import *
import utils.dataset as dataset
from models.yolov3 import YoloLoss
import neptune


# Initialize Neptune run
run = neptune.init_run(
    project="abdalraheem.ijjeh/ADAS-FLIR-DL-prediction",
    api_token=os.getenv("NEPTUNE_API_TOKEN"),
)


# Neptune callback for logging metrics
class NeptuneCallback(Callback):
    def __init__(self, run_):
        super().__init__()
        self.run = run_

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        for metric_name, value in logs.items():
            self.run[metric_name].log(value)


########################################################################################################################
flags.DEFINE_integer('size', 416, 'the input size for model')
flags.DEFINE_integer('epochs', 1000, 'number of epochs')
flags.DEFINE_integer('batch_size', 4, 'batch size')
flags.DEFINE_integer('num_classes', 1, 'number of classes in the model')
flags.DEFINE_string('classes', 'model_data/adas_flir_classes.txt', 'path to classes file')
flags.DEFINE_integer('yolo_max_boxes', 100, 'maximum number of boxes per image')
flags.DEFINE_string('anchors', 'model_data/yolov3_anchors.txt', 'path to anchors file')
flags.DEFINE_string('train_dataset', 'dataset/images_thermal_train_single_class_person.tfrecord',
                    'path to the train dataset')
flags.DEFINE_string('val_dataset', 'dataset/images_thermal_val_single_class_person.tfrecord',
                    'path to the validation dataset')
flags.DEFINE_boolean('transfer', False, 'Transfer learning or not')
flags.DEFINE_string('pretrained_weights', 'checkpoints/yolov3.weights.h5',
                    'path to pretrained weights file')
flags.DEFINE_integer('weights_num_classes', 1,
                     'specify num class for `weights` file if different, '
                     'useful in transfer learning with different number of classes')
flags.DEFINE_string('output', 'model_data/yolov3.keras',
                    'path to save model')


########################################################################################################################
def global_conv_layer(x, k=1):
    # (1xk) convolution
    conv_1xk = Conv2D(filters=x.shape[-1],
                      kernel_size=(1, k),
                      padding='same',
                      use_bias=False)(x)

    # (kx1) convolution
    conv_kx1 = Conv2D(filters=x.shape[-1],
                      kernel_size=(k, 1),
                      padding='same',
                      use_bias=False)(x)

    # Combine (1xk) and (kx1) convolutions
    combined = Add()([conv_1xk, conv_kx1])

    # Optionally apply (kx1) + (1xk) convolutions as well
    conv_kx1_reversed = Conv2D(filters=x.shape[-1],
                               kernel_size=(k, 1),
                               padding='same',
                               use_bias=False)(combined)

    conv_1xk_reversed = Conv2D(filters=x.shape[-1],
                               kernel_size=(1, k),
                               padding='same',
                               use_bias=False)(combined)

    global_conv_out = Add()([conv_kx1_reversed, conv_1xk_reversed])

    return global_conv_out


def conv_block(x, conv_params, use_gcl=False, k=5, skip=True):
    for param in conv_params:
        x = Conv2D(filters=param['filter'],
                   kernel_size=param['kernel'],
                   strides=param['stride'],
                   padding='same',
                   use_bias=not param['bnorm'])(x)
        if param['bnorm']:
            x = BatchNormalization()(x)
        if param['leaky']:
            x = LeakyReLU(alpha=0.1)(x)

    if use_gcl:
        x = global_conv_layer(x, k=k)

    return x


def make_yolov3_model(num_classes_):
    input_image = Input(shape=(416, 416, 3))

    # Initial layers
    x = conv_block(input_image,
                   [{'filter': 32, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 0},
                    {'filter': 64, 'kernel': 3, 'stride': 2, 'bnorm': True, 'leaky': True, 'layer_idx': 1},
                    {'filter': 32, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 2},
                    {'filter': 64, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 3}])

    # Residual Blocks with GCLs
    x = conv_block(x, [{'filter': 128, 'kernel': 3, 'stride': 2, 'bnorm': True, 'leaky': True, 'layer_idx': 5},
                       {'filter': 64, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 6},
                       {'filter': 128, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 7}],
                   use_gcl=True)

    x = conv_block(x, [{'filter': 64, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 9},
                       {'filter': 128, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 10}],
                   use_gcl=True)

    x = conv_block(x, [{'filter': 256, 'kernel': 3, 'stride': 2, 'bnorm': True, 'leaky': True, 'layer_idx': 12},
                       {'filter': 128, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 13},
                       {'filter': 256, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 14}],
                   use_gcl=True)

    # 7 Residual Blocks with GCLs
    for i in range(7):
        x = conv_block(x, [
            {'filter': 128, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 16 + i * 3},
            {'filter': 256, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 17 + i * 3}])
        x = conv_block(x, [
            {'filter': 128, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 16 + i * 3},
            {'filter': 256, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 17 + i * 3}],
                       use_gcl=True)

    skip_36 = x

    x = conv_block(x, [{'filter': 512, 'kernel': 3, 'stride': 2, 'bnorm': True, 'leaky': True, 'layer_idx': 37},
                       {'filter': 256, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 38},
                       {'filter': 512, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 39}])

    # 7 Residual Blocks with GCLs
    for i in range(7):
        x = conv_block(x, [
            {'filter': 256, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 41 + i * 3},
            {'filter': 512, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 42 + i * 3}])
        x = conv_block(x, [
            {'filter': 256, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 41 + i * 3},
            {'filter': 512, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 42 + i * 3}],
                       use_gcl=True)

    skip_61 = x

    x = conv_block(x, [{'filter': 1024, 'kernel': 3, 'stride': 2, 'bnorm': True, 'leaky': True, 'layer_idx': 62},
                       {'filter': 512, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 63},
                       {'filter': 1024, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 64}])

    # 3 Residual Blocks with GCLs
    for i in range(3):
        x = conv_block(x, [
            {'filter': 512, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 66 + i * 3},
            {'filter': 1024, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 67 + i * 3}])
        x = conv_block(x, [
            {'filter': 512, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 66 + i * 3},
            {'filter': 1024, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 67 + i * 3}],
                       use_gcl=True)

    x = conv_block(x, [{'filter': 512, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 75},
                       {'filter': 1024, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 76},
                       {'filter': 512, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 77},
                       {'filter': 1024, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 78},
                       {'filter': 512, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 79}],
                   skip=False)

    # Detection layers
    yolo_82 = conv_block(x,
                         [{'filter': 1024, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 80},
                          {'filter': 3 * (num_classes_ + 5), 'kernel': 1, 'stride': 1, 'bnorm': False,
                           'leaky': False, 'layer_idx': 81}])

    x = conv_block(x, [{'filter': 256, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 84}],
                   skip=False)
    x = UpSampling2D(2)(x)
    x = concatenate([x, skip_61])

    x = conv_block(x, [{'filter': 256, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 87},
                       {'filter': 512, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 88},
                       {'filter': 256, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 89},
                       {'filter': 512, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 90},
                       {'filter': 256, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 91}],
                   skip=False)

    yolo_94 = conv_block(x, [{'filter': 512, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 92},
                             {'filter': 3 * (num_classes_ + 5), 'kernel': 1, 'stride': 1, 'bnorm': False,
                              'leaky': False, 'layer_idx': 93}])

    x = conv_block(x, [{'filter': 128, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 96}],
                   skip=False)
    x = UpSampling2D(2)(x)
    x = concatenate([x, skip_36])

    yolo_106 = conv_block(
        x,
        [
            {'filter': 128, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 99},
            {'filter': 256, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 100},
            {'filter': 128, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 101},
            {'filter': 256, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 102},
            {'filter': 128, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 103},
            {'filter': 256, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 104},
            {'filter': 3 * (num_classes_ + 5), 'kernel': 1, 'stride': 1, 'bnorm': False, 'leaky': False,
             'layer_idx': 105}
        ],
        skip=False)

    model = Model(input_image, [yolo_82, yolo_94, yolo_106])
    return model


def train(argv):
    # Load the tf.dataset.Dataset from TFRecord files
    raw_train_ds = dataset.load_tfrecord_dataset(FLAGS.train_dataset)
    raw_val_ds = dataset.load_tfrecord_dataset(FLAGS.val_dataset)

    # Preprocess the dataset
    anchors = np.array(
        [
            (8, 19), (17, 44), (29, 75),
            (45, 119), (65, 176), (95, 255),
            (142, 375), (210, 553), (353, 852)], np.float32)
    anchors = np.round(anchors).astype(int)
    print(anchors)
    anchors_normalized = anchors / FLAGS.size  # for build_target
    anchor_masks = np.array([[6, 7, 8],
                             [3, 4, 5],
                             [0, 1, 2]])

    train_ds = raw_train_ds.map(lambda x, y: (
        dataset.preprocess_data(
            x, y,
            anchors_normalized, anchor_masks,
            image_size=FLAGS.size,
            yolo_max_boxes=FLAGS.yolo_max_boxes)
    ))
    # visualize.visualize_raw_data(train_ds, class_names, n=3)
    train_ds = train_ds.shuffle(buffer_size=512).batch(FLAGS.batch_size).repeat()
    train_ds = train_ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    # Sample a single batch from train_ds
    for batch in train_ds.take(1):
        x, y = batch  # x is the batch of images, y is the batch of labels

        # Print the shapes of x and y
        print("x shape:", x.shape)
        for i, y_i in enumerate(y):
            print(f"y[{i}] shape:", y_i.shape)

        # Optionally print out a sample from x and y to inspect them
        print("Sample x:", x[0].numpy())
        # for i, y_i in enumerate(y):
        #     print(f"Sample y[{i}]:", y_i[0].numpy())
    val_ds = raw_val_ds.map(lambda x, y: (
        dataset.preprocess_data(
            x, y,
            anchors_normalized, anchor_masks,
            image_size=FLAGS.size,
            yolo_max_boxes=FLAGS.yolo_max_boxes)
    ))
    val_ds = val_ds.batch(FLAGS.batch_size).repeat()
    ####################################################################################################################
    # Check shapes and contents of train_ds and val_ds
    for images, labels in train_ds.take(1):  # Take only one batch
        print("Train Dataset - Batch Shape:")
        print("Images shape:", images.shape)  # Shape of images
        print("Labels shape:")
        for i, label in enumerate(labels):
            print(f"Label {i} shape:", label.shape)  # Shape of each label element
            # Iterate through each sample in the batch
            for j in range(label.shape[0]):
                non_zero_values = label.numpy()[j][label.numpy()[j] != 0]  # Get non-zero values for sample j
                print(f"Label {i} example {j} (non-zero values):", non_zero_values)

    for images, labels in val_ds.take(1):  # Take only one batch
        print("Validation Dataset - Batch Shape:")
        print("Images shape:", images.shape)  # Shape of images
        print("Labels shape:")
        for i, label in enumerate(labels):
            print(f"Label {i} shape:", label.shape)  # Shape of each label element
            # Iterate through each sample in the batch
            for j in range(label.shape[0]):
                non_zero_values = label.numpy()[j][label.numpy()[j] != 0]  # Get non-zero values for sample j
                print(f"Label {i} example {j} (non-zero values):", non_zero_values)
    #
    # ####################################################################################################################
    num_classes = 1
    model = make_yolov3_model(num_classes)
    model.summary()
    # Compile the model
    optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.001)
    yolo_loss = [YoloLoss(anchors[mask], classes=FLAGS.num_classes)
                 for mask in anchor_masks]
    ####################################################################################################################
    # metrics_list = [
    #     [Precision(), Recall(), F1Score()] for _ in range(3)
    # ]
    ####################################################################################################################
    model.summary()
    model.compile(optimizer=optimizer,
                  loss=yolo_loss,
                  ########################
                  # metrics=metrics_list,
                  ########################
                  run_eagerly=False)

    # Callbacks
    callbacks = [
        # ReduceLROnPlateau(verbose=1),
        ModelCheckpoint('trained_models/detection_models/checkpoints/yolov3_train_{epoch}_TIR.keras',
                        monitor='val_loss',
                        save_best_only=True,
                        ),  # verbose=1, save_weights_only=True
        TensorBoard(log_dir='logs'),
        EarlyStopping(monitor='val_loss', patience=1000, restore_best_weights=True),
        NeptuneCallback(run_=run)

    ]
    steps_per_epoch = math.ceil(20531 / FLAGS.batch_size)
    steps_per_epoch_val = math.ceil(2197 / FLAGS.batch_size)
    model.fit(train_ds,
              epochs=FLAGS.epochs,
              callbacks=callbacks,
              validation_data=val_ds,
              validation_steps=steps_per_epoch_val,
              steps_per_epoch=steps_per_epoch
              )

    # Save the model
    model.save("trained_models/detection_models/Yolo_GCN_model_TIR.keras")
    run.stop()


if __name__ == "__main__":
    app.run(train)
