"""
Author: Abdalraheem A. Ijjeh
Affiliation: Postdoc researcher / GRVC Robotics Laboratory, University of Seville, 41092 Seville, Spain

Description:
    This script processes images and their associated ground truth and prediction data.
    It calculates the Intersection over Union (IoU) between ground truth boxes and
    predicted boxes, visualizes the results by drawing bounding boxes on the images,
    and logs the evaluation metrics including True Positives (TP), False Positives (FP),
    and False Negatives (FN) to a log file.

    The script includes the following steps:
    - Reads and parses ground truth and prediction data from JSON files.
    - Calculates IoU to match predictions with ground truth.
    - Draws bounding boxes and labels on the images.
    - Logs the results including IoU, confidence, and TP/FP/FN counts.
    - Saves the annotated images to specified output directories.

    Key Features:
    - IoU calculation for object detection evaluation.
    - Visualization of bounding boxes and confidence scores.
    - Logging of evaluation metrics.

Usage:
    1. Update the `data_dir`, `output_dir`, and `log_file` paths in the `process_directory` call.
    2. Run the script to process the images and generate annotated images and logs.

Requirements:
    - OpenCV
    - JSON
    - Glob
    - OS
"""

import json
import cv2
import os
import glob


def calculate_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)

    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    iou = interArea / float(boxAArea + boxBArea - interArea)

    return iou


def draw_bounding_boxes(image_path, ground_truth_path, prediction_paths, output_dir, log_file):
    image_T = cv2.imread(image_path)
    image_path_2 = image_path.split("/")
    image_path_2[-1] = str(image_path_2[-1]).replace('T.jpg', 'W.jpg')
    image_path_2 = '/'.join(image_path_2)
    image_W = cv2.imread(image_path_2)

    # Load ground truth dataset
    with open(ground_truth_path, 'r') as f:
        ground_truth_data = json.load(f)

    ground_truth_boxes = []
    for shape in ground_truth_data['shapes']:
        if shape['shape_type'] == 'rectangle':
            x1, y1 = shape['points'][0]
            x2, y2 = shape['points'][1]
            ground_truth_boxes.append({
                'label': shape['label'],
                'box': [x1, y1, x2, y2],
                'matched': False
            })
            cv2.rectangle(image_T, (int(x1), int(y1)), (int(x2), int(y2)), (255, 255, 255), 2)  # White for GT boxes
            cv2.rectangle(image_W, (int(x1), int(y1)), (int(x2), int(y2)), (255, 255, 255), 2)  # White for GT boxes

    model_colors = {
        'efficientdet_d5': (255, 0, 0),  # Blue
        'efficientdet_d6': (0, 255, 0),  # Green
        'efficientdet_d7': (0, 0, 255),  # Red
        'faster_rcnn_inception_resnet_v2': (255, 255, 0),  # Cyan
        'faster_rcnn_resnet50_v1': (255, 0, 255),  # Magenta
        'faster_rcnn_resnet101_v1': (0, 255, 255),  # Yellow
        'ssd_mobilenet_v2_fpnlite': (128, 0, 128),  # Purple
        'ssd_resnet50_v1_fpn': (0, 128, 128),  # Teal
        'YOLO_GCN': (128, 128, 0),  # Olive
        'centernet_hg104': (255, 165, 0)  # Orange
    }

    # Open the log file for appending results
    with open(log_file, 'a') as log:
        # Iterate over each model prediction file
        for pred_path in prediction_paths:
            model_name = os.path.basename(pred_path).split('_Pred_')[0]
            model_name = model_name.split('_T_')[-1]
            print(f"Processing {model_name} for {os.path.basename(image_path)}")
            model_output_dir = os.path.join(output_dir, model_name)
            model_color = model_colors.get(model_name,
                                           (255, 255, 255))  # Default color is white if model name not found

            if not os.path.exists(model_output_dir):
                os.makedirs(model_output_dir)

            # Copy image to a new directory to avoid overwriting
            image_copy_T = image_T.copy()
            image_copy_W = image_W.copy()

            with open(pred_path, 'r') as f:
                prediction_data = json.load(f)

            # Counters for TP, FN, and FP
            tp_count = 0
            fn_count = 0
            fp_count = 0

            # Create a set to keep track of ground truth boxes that have been matched
            matched_gt_boxes = set()

            for pred_shape in prediction_data['shapes']:
                x1, y1 = pred_shape['points'][0]
                x2, y2 = pred_shape['points'][1]
                pred_box = [x1, y1, x2, y2]
                pred_label = pred_shape['label']
                pred_confidence = pred_shape.get('confidence', None)

                matched_gt_box = None
                max_iou = 0

                # Check for matching ground truth boxes
                for gt_shape in ground_truth_boxes:
                    if gt_shape['label'] == pred_label and not gt_shape['matched']:
                        iou = calculate_iou(gt_shape['box'], pred_box)
                        if iou > max_iou:
                            max_iou = iou
                            matched_gt_box = gt_shape

                # Discard predicted boxes with IoU < 0.5
                if max_iou >= 0.5:
                    cv2.rectangle(image_copy_T, (int(x1), int(y1)), (int(x2), int(y2)), model_color, 2)
                    cv2.rectangle(image_copy_W, (int(x1), int(y1)), (int(x2), int(y2)), model_color, 2)
                    text_iou = f"IoU: {max_iou:.2f}"
                    if pred_confidence is not None:
                        text_confidence = f"Confidence: {pred_confidence:.2f}"
                        cv2.putText(image_copy_T, text_confidence, (int(x1), int(y1) + 10), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.5,
                                    model_color, 2)
                        cv2.putText(image_copy_W, text_confidence, (int(x1), int(y1) + 10), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.5,
                                    model_color, 2)

                    cv2.putText(image_copy_T, text_iou, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                model_color, 2)
                    cv2.putText(image_copy_W, text_iou, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                model_color, 2)

                    # Log the details
                    log.write(f"{os.path.basename(image_path)}\t{model_name}\t{pred_label}\t{max_iou:.2f}\t"
                              f"{pred_confidence if pred_confidence is not None else 'N/A'}\n")

                    # Update TP and mark the ground truth as matched
                    tp_count += 1
                    matched_gt_box['matched'] = True
                    matched_gt_boxes.add(tuple(matched_gt_box['box']))
                else:
                    # False Positive if no match or low IoU
                    fp_count += 1
                    log.write(f"{os.path.basename(image_path)}\t{model_name}\t{pred_label}\tFP\t"
                              f"{pred_confidence if pred_confidence is not None else 'N/A'}\n")

            # Count False Negatives for ground truth boxes not matched
            for gt_shape in ground_truth_boxes:
                if not gt_shape['matched']:
                    fn_count += 1
                    log.write(f"{os.path.basename(image_path)}\t{model_name}\t{gt_shape['label']}\tFN\n")

            # Log TP, FP, FN counts
            log.write(f"Summary for {os.path.basename(image_path)}:\n")
            log.write(f"Model: {model_name}\tTP: {tp_count}\tFP: {fp_count}\tFN: {fn_count}\n")
            log.write("---------------------------------------------------------\n")

            # Save the resulting image
            output_image_name_T = os.path.basename(image_path)
            output_image_name_W = os.path.basename(image_path_2)

            output_path_T = os.path.join(model_output_dir, output_image_name_T)
            output_path_W = os.path.join(model_output_dir, output_image_name_W)

            cv2.imwrite(output_path_T, image_copy_T)
            cv2.imwrite(output_path_W, image_copy_W)


def process_directory(data_dir, output_dir, log_file):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Find all images in the directory
    image_paths = glob.glob(os.path.join(data_dir, "*.jpg"))

    for image_path in sorted(image_paths):
        image_name = os.path.splitext(os.path.basename(image_path))[0]
        ground_truth_path = os.path.join(data_dir, f"{image_name}.json")

        if not os.path.exists(ground_truth_path):
            print(f"Ground truth for {image_name} not found. Skipping.")
            continue

        # Find all prediction files for this image
        prediction_paths = glob.glob(os.path.join(data_dir, f"{image_name}_*_Pred_.json"))

        if not prediction_paths:
            print(f"No predictions found for {image_name}. Skipping.")
            continue

        draw_bounding_boxes(image_path, ground_truth_path, prediction_paths, output_dir, log_file)


process_directory(
    "/home/abdalraheem/Documents/GitHub/TEMA_Visual_IR_DL_Detection/datasets/Smoke_dataset/DJI_202407230824_030"
    "/valid_thermal_images",
    "/home/abdalraheem/Documents/GitHub/TEMA_Visual_IR_DL_Detection/datasets/Smoke_dataset/DJI_202407230824_030"
    "/valid_thermal_images/output_1",
    "/home/abdalraheem/Documents/GitHub/TEMA_Visual_IR_DL_Detection/datasets/Smoke_dataset/DJI_202407230824_030"
    "/day_1_log.txt"
)
