"""
Author: Abdalraheem A. Ijjeh
Affiliation: Postdoc researcher / GRVC Robotics Laboratory, University of Seville, 41092 Seville, Spain

Description:
    This script evaluates the memory usage and inference speed of a TensorFlow model.
    It calculates the memory required by the model's weights, measures the inference
    speed, and estimates the memory usage before and after running the model.
    This is useful for profiling and optimizing model performance.

    The script includes the following steps:
    - Measuring initial memory usage.
    - Loading a TensorFlow SavedModel.
    - Calculating the memory used by the model's weights.
    - Measuring the model's inference speed.
    - Performing inference to observe memory changes.
    - Estimating memory used by intermediate activations and buffers.

    Key Features:
    - Memory usage calculation of model weights and intermediate buffers.
    - Inference speed measurement.
    - Handling TensorFlow models and profiling memory.

Usage:
    1. Update the model path in `tf.saved_model.load` to point to your SavedModel directory.
    2. Run the script to print memory usage and inference speed metrics for the model.

Requirements:
    - TensorFlow
    - NumPy
    - psutil (for memory usage)
    - gc (for garbage collection)
    - os
    - time
"""

import tensorflow as tf
import numpy as np
import gc
import os
import time
import psutil


def get_memory_usage():
    """Get memory usage of the process."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 ** 2)  # Memory usage in MB


def calculate_model_memory(concrete_function):
    """Calculate the memory used by the model's weights."""
    total_parameters = 0
    total_memory = 0

    for variable in concrete_function.variables:
        total_parameters += np.prod(variable.shape)
        total_memory += variable.numpy().nbytes

    total_memory_mb = total_memory / (1024 ** 2)
    total_parameters_million = total_parameters / 1e6
    return total_parameters_million, total_memory_mb


def calculate_inference_speed(model, input_tensor):
    """Calculate the inference speed of the model."""
    for _ in range(5):  # Warm-up runs
        _ = model(input_tensor)

    start_time = time.time()
    _ = model(input_tensor)
    end_time = time.time()

    inference_time_ms = (end_time - start_time) * 1000
    return inference_time_ms


def run_model():
    try:
        initial_memory = get_memory_usage()
        model = tf.saved_model.load('tf_models_architectures/centernet_hg104/saved_model')
        infer = model.signatures['serving_default']
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    total_parameters, model_memory_mb = calculate_model_memory(infer)
    print(f"Model Weights Memory: {model_memory_mb:.2f} MB")
    print(f"Total Parameters: {total_parameters:.2f} M")

    input_tensor = tf.random.uniform(shape=(1, 416, 416, 3), dtype=tf.float32)
    input_tensor_uint8 = tf.cast(input_tensor * 255, tf.uint8)

    inference_speed_ms = calculate_inference_speed(infer, input_tensor_uint8)
    print(f"Inference Speed: {inference_speed_ms:.2f} ms")

    _ = infer(input_tensor_uint8)

    gc.collect()
    final_memory = get_memory_usage()
    total_memory_usage = final_memory - initial_memory

    activations_and_buffers_memory_mb = total_memory_usage - model_memory_mb

    print(f"Memory used after loading and inference: {total_memory_usage:.2f} MB")
    print(f"Intermediate Activations and Buffers Memory: {activations_and_buffers_memory_mb:.2f} MB")
    print("Power Consumption: Measurement requires external tools.")


if __name__ == "__main__":
    run_model()
