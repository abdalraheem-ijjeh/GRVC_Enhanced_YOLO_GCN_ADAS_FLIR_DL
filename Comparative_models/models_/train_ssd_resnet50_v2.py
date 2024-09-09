"""
Author: Abdalraheem A. Ijjeh
Affiliation: Postdoc researcher / GRVC Robotics Laboratory, University of Seville, 41092 Seville, Spain

Description:
    This script is used to fine-tune the SSD ResNet50 v1 model using TensorFlow Object Detection API used for
    Person detection in smoky environments.
    It modifies the pipeline configuration for custom dataset paths, sets hyperparameters such as batch size,
    number of epochs, optimizer, and adjusts the loss function weights.

    Key Features:
    - Loads and modifies the pipeline configuration.
    - Configures batch size, number of epochs, and total training steps.
    - Sets Adam optimizer with a constant learning rate.
    - Adjusts classification and localization loss weights.
    - Trains the model using TensorFlow's object detection model library.

Usage:
    To execute this script, ensure that TensorFlow Object Detection API is set up and the required
    dataset (in TFRecord format) and label map are available.
    Execute the script via a Python environment with TensorFlow installed:

    ```
    python train_ssd_resnet50_v1.py
    ```

Paths:
    - MODEL_DIR: Path where the pre-trained model and checkpoint are stored.
    - PIPELINE_CONFIG_PATH: Path to the pipeline configuration file (pipeline.config).
    - OUTPUT_DIR: Directory where model checkpoints and logs will be saved.
    - TFRECORD_PATH: Path to your dataset in TFRecord format.
    - LABEL_MAP_PATH: Path to the label map file (label_map.pbtxt).

Hyperparameters:
    - BATCH_SIZE: Number of samples per batch.
    - NUM_EPOCHS: Number of epochs for training.
    - NUM_TRAINING_SAMPLES: Total number of training samples in the dataset.
    - LEARNING_RATE: Learning rate for the optimizer.
"""

import sys
import os

import tensorflow as tf
from object_detection import model_lib
from object_detection.utils import config_util
from google.protobuf import text_format
from object_detection.protos import pipeline_pb2

# Paths
MODEL_DIR = "Comparative_models/models_/tf_models_architectures/ssd_resnet50_v1_fpn"
PIPELINE_CONFIG_PATH = "pipeline.config"
OUTPUT_DIR = "Comparative_models/models_/tf_models_architectures/ssd_resnet50_v1_fpn/checkpoint"
TFRECORD_PATH = "dataset"
LABEL_MAP_PATH = "Comparative_models/label_map.pbtxt"

# Checkpoint from the pre-trained model
CHECKPOINT_PATH = os.path.join(MODEL_DIR, "checkpoint")


# Modify the pipeline configuration to match your dataset
def modify_pipeline_config(config_path, tfrecord_path, label_map_path, checkpoint_path):
    """
    Modify the pipeline configuration to match the dataset and checkpoint.

    Args:
    config_path (str): Path to the original pipeline configuration file.
    tfrecord_path (str): Path to the directory containing TFRecord files.
    label_map_path (str): Path to the label map file.
    checkpoint_path (str): Path to the pre-trained model checkpoint directory.
    """
    # Load and parse the pipeline config file
    pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()
    with tf.io.gfile.GFile(config_path, "r") as f:
        proto_str = f.read()
        text_format.Merge(proto_str, pipeline_config)

    # Update the number of classes to 1 (for "person" class)
    pipeline_config.model.ssd.num_classes = 1
    pipeline_config.train_config.fine_tune_checkpoint = os.path.join(checkpoint_path, 'ckpt-1')
    pipeline_config.train_config.fine_tune_checkpoint_type = "detection"

    # Set the paths for the train and eval TFRecord files
    pipeline_config.train_input_reader.tf_record_input_reader.input_path[:] = [
        os.path.join(tfrecord_path, "images_thermal_train_single_class_person.tfrecord")]
    pipeline_config.train_input_reader.label_map_path = label_map_path

    pipeline_config.eval_input_reader[0].tf_record_input_reader.input_path[:] = [
        os.path.join(tfrecord_path, "images_thermal_val_single_class_person.tfrecord")]
    pipeline_config.eval_input_reader[0].label_map_path = label_map_path

    # Set batch size, optimizer, and training steps
    pipeline_config.train_config.batch_size = 32  # Batch size
    pipeline_config.train_config.num_steps = 1000  # Training steps (epochs equivalent)

    # Optimizer (Momentum)
    pipeline_config.train_config.optimizer.momentum_optimizer.learning_rate.constant_learning_rate.learning_rate = 0.001
    pipeline_config.train_config.optimizer.momentum_optimizer.momentum_optimizer_value = 0.9

    # Loss function adjustments (weight for classification and localization loss)
    pipeline_config.model.ssd.loss.classification_loss.weight = 1.0
    pipeline_config.model.ssd.loss.localization_loss.weight = 1.0

    # Save the updated config back to disk
    config_text = text_format.MessageToString(pipeline_config)
    with tf.io.gfile.GFile(os.path.join(OUTPUT_DIR, 'pipeline.config'), "wb") as f:
        f.write(config_text)


# Train the model
def train_model():
    """
    Initiates the model training process.

    The pipeline configuration is first modified with custom hyperparameters,
    dataset paths, and checkpoint paths. Then, the TensorFlow Object Detection
    API's training loop is invoked using the modified pipeline configuration.
    """
    # Modify the pipeline configuration
    modify_pipeline_config(PIPELINE_CONFIG_PATH, TFRECORD_PATH, LABEL_MAP_PATH, CHECKPOINT_PATH)

    # Start the training process
    model_lib.train_loop(
        pipeline_config_path=os.path.join(OUTPUT_DIR, 'pipeline.config'),
        model_dir=OUTPUT_DIR,
        train_steps=None,  # Train indefinitely until manually stopped
        use_tpu=False
    )


if __name__ == "__main__":
    train_model()
