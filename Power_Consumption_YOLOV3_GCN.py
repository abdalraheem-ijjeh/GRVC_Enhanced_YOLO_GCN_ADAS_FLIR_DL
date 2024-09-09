"""
Author: Abdalraheem A. Ijjeh
Affiliation: Postdoc researcher / GRVC Robotics Laboratory, University of Seville, 41092 Seville, Spain

Description:
This script evaluates the performance of a Keras model by measuring inference time, computing memory requirements,
 and logging GPU power consumption.
  It is designed to work with Keras models and requires NVIDIA GPUs for power consumption logging.

The script performs the following tasks:
- Loads a Keras model from a specified file path.
- Measures the inference time of the model.
- Computes memory requirements of the model parameters.
- Logs GPU power consumption using `nvidia-smi`.
- Reads and computes the average GPU power draw from the log file.

Key Features:
- Inference Time Measurement: Calculates the time taken by the model to perform inference in milliseconds.
- Memory Requirements Calculation: Estimates the memory required to store the model parameters.
- Power Consumption Logging: Logs and computes the average power draw of the GPU during inference.

Requirements:
- TensorFlow: For loading and using the Keras model.
- NumPy: For creating and manipulating arrays.
- Pandas: For reading and processing power consumption logs.
- OS: For executing system commands.
- re: For regular expression operations.

Usage:
1. Model Path: Specify the path to the Keras model file using `model_path`.
2. Inference Measurement: Automatically measures inference time with a random input tensor.
3. Power Consumption Logging: Starts and stops logging GPU power consumption.
4. Memory Requirements Calculation: Computes and prints memory requirements.

Example Command:
```bash
python script.py
```

Notes:
- Ensure that `nvidia-smi` is installed and properly configured on your system to enable GPU power consumption logging.
- The script assumes that the model file is in Keras format (.keras) and can be loaded directly
 using `tf.keras.models.load_model`.

"""

import tensorflow as tf
import numpy as np
import time
import pandas as pd
import os
import re


def load_model(model_path):
    """
    Loads a Keras model from the specified file path.
    Parameters:
        - `model_path` (str): Path to the Keras model file.
    Returns:
        - The loaded Keras model.
    """

    """Load a Keras model from the specified file path."""
    return tf.keras.models.load_model(model_path, compile=False)


def measure_inference_time(model, input_tensor):
    """
    Measures the inference time of the model.
    Parameters:
        - `model`: The loaded Keras model.
        - `input_tensor` (tf.Tensor): The input tensor for the model.
    Returns:
        - Inference time in milliseconds.
    """
    """Calculate the inference speed of the model."""
    # Warm-up runs
    for _ in range(50):
        start_time = time.time()
        _ = model.predict(input_tensor)
        end_time = time.time()
        inference_time_ms = (end_time - start_time) * 1000  # Convert to milliseconds

    start_time = time.time()
    _ = model.predict(input_tensor)
    end_time = time.time()

    inference_time_ms = (end_time - start_time) * 1000  # Convert to milliseconds
    return inference_time_ms


def compute_memory_requirements(model):
    """
    Computes the memory required to store the model parameters.
    Parameters:
        - `model`: The loaded Keras model.
    Returns:
        - Memory required in MB, or `None` if an error occurs.
    """
    """Compute the memory required to store the model parameters."""
    try:
        total_params = 0
        for variable in model.trainable_variables:
            total_params += np.prod(variable.shape)
        memory_mb = total_params * 4 / (1024 ** 2)  # 4 bytes per parameter for 32-bit floats
        return memory_mb
    except Exception as e:
        print(f"Error calculating memory requirements: {e}")
        return None


def start_power_logging():
    """
    Starts logging GPU power consumption using `nvidia-smi`.
    Returns:
        - Path to the power log file.
    """

    """Start logging GPU power consumption."""
    power_log_file = 'power_usage_log.csv'
    print("Starting power logging...")
    power_log_command = f"nvidia-smi --query-gpu=timestamp,power.draw --format=csv -l 1 > {power_log_file} &"
    os.system(power_log_command)
    return power_log_file


def stop_power_logging():
    """
    Stops logging GPU power consumption.
    """
    """Stop logging GPU power consumption."""
    print("Stopping power logging...")
    os.system("pkill -f 'nvidia-smi --query-gpu=timestamp,power.draw'")


def read_power_log(power_log_file):
    """
    Reads and computes the average GPU power draw from the log file.
    Parameters:
        - `power_log_file` (str): Path to the power log file.
    Returns:
        - Average power draw in watts, or `None` if the log file is not found.

    """
    """Read and compute average GPU power draw from the log file."""
    try:
        power_data = pd.read_csv(power_log_file)
        power_draw_column = power_data[' power.draw [W]']
        cleaned_power_draw = power_draw_column.apply(lambda x: re.sub(r'[^\d.]', '', x))
        power_draw_numeric = pd.to_numeric(cleaned_power_draw, errors='coerce')
        average_power_draw = power_draw_numeric.mean()
        return average_power_draw
    except FileNotFoundError:
        print("Power usage log file not found.")
        return None


def main():
    """
    Main function to execute the performance evaluation script.
    Loads the model, measures inference time, logs power consumption,
    computes power usage, and reports memory requirements.
    """
    model_path = 'trained_models/detection_models/checkpoints/yolov3_train_1_TIR.keras'
    # model_path = 'Yolo_GCN_model.keras'
    model = load_model(model_path)

    # Define input tensor with shape (1, 416, 416, 3) and uint8 type
    input_tensor = tf.convert_to_tensor(np.random.randint(0, 256, (1, 416, 416, 3), dtype=np.uint8), dtype=tf.uint8)

    # Start logging power consumption
    power_log_file = start_power_logging()

    # Measure inference time
    inference_time = measure_inference_time(model, input_tensor)
    print(f"Inference time on GPU: {inference_time:.4f} milliseconds")

    # Stop logging power consumption
    stop_power_logging()

    # Read and report power usage
    average_power_draw = read_power_log(power_log_file)
    if average_power_draw is not None:
        print(f"Average GPU Power Draw: {average_power_draw:.2f} W")

    # Compute and report memory requirements
    memory_mb = compute_memory_requirements(model)
    if memory_mb is not None:
        print(f"Memory required for model parameters: {memory_mb:.2f} MB")


if __name__ == "__main__":
    main()
