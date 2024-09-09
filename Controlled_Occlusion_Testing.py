"""
Author: Abdalraheem A. Ijjeh
Affiliation: Postdoc researcher / GRVC Robotics Laboratory, University of Seville, 41092 Seville, Spain

Description:
    This script processes images based on JSON annotations to apply occlusions to specific regions.
    It reads bounding box annotations from JSON files, applies occlusions to these bounding boxes
    in the corresponding images, and saves the occluded images to a specified directory.

    The script includes the following steps:
    - Reading bounding box annotations from JSON files.
    - Adding occlusion to the bounding box regions in the images.
    - Saving the resulting occluded images.

    Key Features:
    - Reading and parsing JSON annotations to extract bounding boxes.
    - Adding occlusions (black rectangles) to bounding box regions in images.
    - Saving the processed images with occlusions.

Usage:
    1. Set `json_dir` to the directory containing your JSON annotation files.
    2. Set `image_dir` to the directory containing the corresponding images.
    3. Specify the `output_dir` where you want to save the occluded images.
    4. Set the `percentage` for occlusion (e.g., 0.2 for 20% occlusion).
    5. Run the script to process and save the occluded images.

Requirements:
    - OpenCV (cv2)
    - NumPy
    - JSON
    - OS
"""

import json
import cv2
import numpy as np
import os


def read_annotations(json_file):
    """
    Reads annotations from a JSON file.

    Args:
        json_file (str): Path to the JSON annotation file.

    Returns:
        dict: A dictionary containing image path, height, width, and annotations.
    """
    with open(json_file, 'r') as f:
        data = json.load(f)

    image_info = {
        "imagePath": data["imagePath"],
        "imageHeight": data["imageHeight"],
        "imageWidth": data["imageWidth"],
        "annotations": []
    }

    for shape in data['shapes']:
        if shape['label'] == 'person' and shape['shape_type'] == 'rectangle':
            x1, y1 = shape['points'][0]
            x2, y2 = shape['points'][1]
            image_info["annotations"].append({
                "bbox": [x1, y1, x2, y2]
            })

    return image_info


def add_occlusion_to_bbox(image, bbox, occlusion_percentage):
    """
    Adds occlusion to a specific bounding box region in the image based on a percentage of the bbox size.

    Args:
        image (numpy array): Input image.
        bbox (list): Bounding box coordinates [x1, y1, x2, y2].
        occlusion_percentage (float): Percentage of the bounding box area to occlude (e.g., 0.2 for 20%).

    Returns:
        numpy array: Image with occlusion added to the bounding box region.
    """
    x1, y1, x2, y2 = map(int, bbox)
    img_with_occlusion = image.copy()

    # Calculate width and height of the bounding box
    bbox_width = x2 - x1
    bbox_height = y2 - y1

    # Calculate occlusion size based on the percentage
    occlusion_width = int(bbox_width * occlusion_percentage)
    occlusion_height = int(bbox_height * occlusion_percentage)

    # Calculate occlusion position inside the bounding box
    occlusion_x = x1 + (bbox_width // 2 - occlusion_width // 2)
    occlusion_y = y1 + (bbox_height // 2 - occlusion_height // 2)

    # Ensure occlusion does not go outside the image boundaries
    occlusion_x = max(x1, occlusion_x)
    occlusion_y = max(y1, occlusion_y)
    occlusion_x_end = min(x2, occlusion_x + occlusion_width)
    occlusion_y_end = min(y2, occlusion_y + occlusion_height)

    # Draw the occlusion (black rectangle)
    cv2.rectangle(img_with_occlusion,
                  (occlusion_x, occlusion_y),
                  (occlusion_x_end, occlusion_y_end),
                  (0, 0, 0),
                  -1)

    return img_with_occlusion


def save_occluded_image(image, output_path, filename):
    """
    Saves the occluded image to the specified output path.

    Args:
        image (numpy array): Image with occlusion added.
        output_path (str): Directory path to save the occluded image.
        filename (str): The name of the file to save.
    """
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    cv2.imwrite(os.path.join(output_path, filename), image)


def process_images(json_dir, image_dir, output_dir, percentage):
    """
    Process all JSON annotation files, apply occlusions, and save the occluded images.

    Args:
        json_dir (str): Directory containing JSON annotation files.
        image_dir (str): Directory containing the images.
        output_dir (str): Directory to save the occluded images.
    """
    # Get all JSON files in the annotation directory
    json_files = [f for f in os.listdir(json_dir) if f.endswith('.json')]

    for json_file in json_files:
        json_path = os.path.join(json_dir, json_file)

        # Read annotation dataset
        annotations = read_annotations(json_path)

        # Load the corresponding image
        image_path = os.path.join(image_dir, annotations["imagePath"])
        image = cv2.imread(image_path)

        if image is None:
            print(f"Image {annotations['imagePath']} not found. Skipping.")
            continue

        # Copy the image to apply occlusions
        occluded_image = image.copy()

        # Apply occlusions to each bounding box
        for idx, ann in enumerate(annotations["annotations"]):
            occluded_image = add_occlusion_to_bbox(occluded_image, ann["bbox"], occlusion_percentage=percentage)

        # Save the final occluded image with all occlusions
        output_filename = f"{annotations['imagePath']}"
        save_occluded_image(occluded_image, output_dir, output_filename)

        print(f"Processed and saved occluded image for {annotations['imagePath']}.")


# Example usage
if __name__ == "__main__":
    main_dir = '/home/abdalraheem/Documents/GitHub/TEMA_Visual_IR_DL_Detection/datasets/Smoke_dataset/'
    json_directory = (os.path.join(main_dir, 'DJI_202406032323_029/valid_thermal_images'))
    # JSON files
    image_directory = json_directory
    occlusion_percentage = 0.3
    output_directory = (os.path.join(main_dir, f"DJI_202406032323_029/valid_thermal_images/occluded_images_{occlusion_percentage}"))
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    process_images(json_directory, image_directory, output_directory, occlusion_percentage)
