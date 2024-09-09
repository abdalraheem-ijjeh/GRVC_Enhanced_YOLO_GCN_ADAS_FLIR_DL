"""
Author: Abdalraheem A. Ijjeh
Affiliation: Postdoc researcher / GRVC Robotics Laboratory, University of Seville, 41092 Seville, Spain


Description:
This script evaluates the performance of a TensorFlow model by measuring inference time, computing memory requirements, and logging GPU power consumption. It is designed to work with TensorFlow models saved in a directory and requires NVIDIA GPUs for power consumption logging.

The script performs the following tasks:

1. Loads a TensorFlow model from a specified directory.
2. Measures the inference time of the model.
3. Computes memory requirements of the model parameters.
4. Logs GPU power consumption using nvidia-smi.
5. Reads and computes average GPU power draw from the log file.

Key Features:
Inference Time Measurement: Calculates the time taken by the model to perform inference.
Memory Requirements Calculation: Estimates the memory required to store the model parameters.
Power Consumption Logging: Logs and computes the average power draw of the GPU during inference.

Usage:
Model Directory: Specify the path to the saved TensorFlow model using model_dir.
Inference Measurement: Automatically measures inference time with a random input tensor.
Power Consumption Logging: Starts and stops logging GPU power consumption.
Memory Requirements Calculation: Optionally computes and prints memory requirements.

Requirements:
TensorFlow: For loading and using the model.
NumPy: For creating and manipulating arrays.
Pandas: For reading and processing power consumption logs.
OS: For executing system commands.
re: For regular expression operations.

"""
import tensorflow as tf
import numpy as np
import time
import pandas as pd
import os
import re


def load_model(model_dir):
    """Load the TensorFlow model from the specified directory."""
    return tf.saved_model.load(model_dir)


def measure_inference_time(model, input_tensor):
    """Calculate the inference speed of the model."""
    for _ in range(50):  # Warm-up runs
        _ = model(input_tensor)

    start_time = time.time()
    _ = model(input_tensor)
    end_time = time.time()

    inference_time_ms = (end_time - start_time)
    return inference_time_ms


def compute_memory_requirements(model_dir):
    """Compute the memory required to store the model parameters."""
    model = load_model(model_dir)

    # We need to access the checkpoint if the model is not directly accessible
    try:
        # Load the checkpoint
        checkpoint = tf.train.Checkpoint(model=model)
        latest_checkpoint = tf.train.latest_checkpoint(model_dir)
        if latest_checkpoint:
            checkpoint.restore(latest_checkpoint)
        else:
            raise FileNotFoundError("Checkpoint file not found.")

        # Collect parameters from the checkpoint
        total_params = 0
        for variable in model.variables:
            total_params += np.prod(variable.shape)
        memory_mb = total_params * 4 / (1024 ** 2)  # 4 bytes per parameter for 32-bit floats
    except Exception as e:
        print(f"Error calculating memory requirements: {e}")
        memory_mb = None

    return memory_mb


def start_power_logging():
    """Start logging GPU power consumption."""
    power_log_file = 'power_usage_log.csv'
    print("Starting power logging...")
    power_log_command = f"nvidia-smi --query-gpu=timestamp,power.draw --format=csv -l 1 > {power_log_file} &"
    os.system(power_log_command)
    return power_log_file


def stop_power_logging():
    """Stop logging GPU power consumption."""
    print("Stopping power logging...")
    os.system("pkill -f 'nvidia-smi --query-gpu=timestamp,power.draw'")


def read_power_log(power_log_file):
    """Read and compute average GPU power draw from the log file."""
    try:
        # Read the power usage log file
        power_data = pd.read_csv(power_log_file)

        # Extract the power draw column and clean it
        power_draw_column = power_data[' power.draw [W]']
        cleaned_power_draw = power_draw_column.apply(lambda x: re.sub(r'[^\d.]', '', x))

        # Convert to numeric
        power_draw_numeric = pd.to_numeric(cleaned_power_draw, errors='coerce')

        # Calculate average power draw
        average_power_draw = power_draw_numeric.mean()
        return average_power_draw
    except FileNotFoundError:
        print("Power usage log file not found.")
        return None


def main():
    # Adjust the model directory path as needed
    model_dir = 'Comparative_models/models_/tf_models_architectures/efficientdet_d7/saved_model'

    # Load the model
    model = load_model(model_dir)

    # Define input tensor with shape (1, 512, 512, 3) and uint8 type
    input_tensor = tf.convert_to_tensor(np.random.randint(0, 256, (1, 416, 416, 3), dtype=np.uint8), dtype=tf.uint8)

    # Start logging power consumption
    power_log_file = start_power_logging()

    # Measure inference time
    inference_time = measure_inference_time(model, input_tensor)
    print(f"Inference time on GPU: {inference_time:.4f} seconds")

    # Stop logging power consumption
    stop_power_logging()

    # Read and report power usage
    average_power_draw = read_power_log(power_log_file)
    if average_power_draw is not None:
        print(f"Average GPU Power Draw: {average_power_draw:.2f} W")

    # # Compute memory requirements
    # memory_mb = compute_memory_requirements(model_dir)
    # if memory_mb is not None:
    #     print(f"Memory required for model parameters: {memory_mb:.2f} MB")


if __name__ == "__main__":
    main()
