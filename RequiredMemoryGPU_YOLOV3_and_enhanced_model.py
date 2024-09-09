"""
Author: Abdalraheem A. Ijjeh
Affiliation: Postdoc researcher / GRVC Robotics Laboratory, University of Seville, 41092 Seville, Spain

Description: This script evaluates the performance of a TensorFlow Keras model by measuring memory usage, calculating
inference speed, and providing estimates for intermediate activations and buffers. It requires TensorFlow and psutil
libraries for model loading, memory measurement, and performance evaluation.

The script performs the following tasks:
- Loads a Keras model from a specified file path.
- Calculates the memory used by the model's weights and architecture.
- Measures the speed of model inference.
- Estimates the memory used by intermediate activations and buffers.
- Provides a warning if the memory usage seems inconsistent.

Key Features:
- Memory Usage Measurement: Computes the memory used by the model's weights and estimates the memory used by intermediate activations and buffers.
- Inference Speed Calculation: Measures the time taken for model inference in milliseconds.
- Garbage Collection: Utilizes garbage collection to get accurate memory usage after model inference.

Requirements:
- TensorFlow: For loading and using the Keras model.
- NumPy: For numerical operations and memory calculations.
- psutil: For measuring the memory usage of the process.
- gc: For garbage collection.
- os: For process and system operations.
- time: For timing the inference process.

Usage:
1. Model Path: Ensure that the Keras model file (`Yolo_GCN_model.keras`) is available at the specified path.
2. Memory Calculation: Computes the memory used by the model weights and estimates additional memory used during inference.
3. Inference Speed Measurement: Measures the time required for model inference in milliseconds.
4. Garbage Collection: Forces garbage collection to get accurate memory usage metrics.

Example Command:
```python
python script.py
```

Notes: - Ensure that TensorFlow, NumPy, and psutil are installed in your environment. - Adjust the model file path
and input tensor shape according to your specific model and requirements. - The script assumes that the model file is
in a compatible Keras format and can be loaded directly using `tf.keras.models.load_model`. - Power consumption
measurement is not included in this script and would require additional external tools.

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


def calculate_model_memory(model):
    """Calculate the memory used by the model's weights."""
    total_parameters = 0
    total_memory = 0

    for variable in model.trainable_variables:
        total_parameters += np.prod(variable.shape)
        total_memory += variable.numpy().nbytes

    total_memory_mb = total_memory / (1024 ** 2)
    total_parameters_million = total_parameters / 1e6
    return total_parameters_million, total_memory_mb


def calculate_inference_speed(model, input_tensor):
    """Calculate the inference speed of the model."""
    # Warm-up runs
    for _ in range(5):
        _ = model(input_tensor)

    # Time the inference
    start_time = time.time()
    _ = model(input_tensor)
    end_time = time.time()

    # Calculate inference time in milliseconds
    inference_time_ms = (end_time - start_time) * 1000
    return inference_time_ms


def run_model():
    try:
        initial_memory = get_memory_usage()

        # Load the Keras model
        model = tf.keras.models.load_model('Yolo_GCN_model.keras',
                                           compile=False)

        # Calculate memory for model weights and architecture
        total_parameters, model_memory_mb = calculate_model_memory(model)
        print(f"Model Weights Memory: {model_memory_mb:.2f} MB")
        print(f"Total Parameters: {total_parameters:.2f} M")

        # Create a dummy input tensor with the correct dataset type (float32)
        input_tensor = tf.random.uniform(shape=(1, 416, 416, 3), dtype=tf.float32)

        # Calculate inference speed
        inference_speed_ms = calculate_inference_speed(model, input_tensor)
        print(f"Inference Speed: {inference_speed_ms:.2f} ms")

        # Perform inference
        _ = model(input_tensor)

        # Force garbage collection to get accurate memory usage
        gc.collect()
        final_memory = get_memory_usage()
        total_memory_usage = final_memory - initial_memory

        # Estimate Intermediate Activations and Buffers Memory
        # Ensure that this calculation is valid and model_memory_mb is realistic
        if total_memory_usage < model_memory_mb:
            print("Warning: Total memory usage is less than model memory. This might indicate an issue.")
            activations_and_buffers_memory_mb = 0
        else:
            activations_and_buffers_memory_mb = total_memory_usage - model_memory_mb

        print(f"Memory used after loading and inference: {total_memory_usage:.2f} MB")
        print(f"Intermediate Activations and Buffers Memory: {activations_and_buffers_memory_mb:.2f} MB")
        print("Power Consumption: Measurement requires external tools.")

    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    run_model()
