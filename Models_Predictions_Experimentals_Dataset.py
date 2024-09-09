"""
Author: Abdalraheem A. Ijjeh
Affiliation: Postdoc researcher / GRVC Robotics Laboratory, University of Seville, 41092 Seville, Spain

Description:
This script performs person detection using a pre-trained model. It supports detection on images, directories of images, and video files. The script processes images, calculates detection metrics, and generates visual output such as annotated images and confusion matrices. Key functionalities include detection confidence filtering, bounding box annotation, and evaluation metrics computation.

The script performs the following tasks:
- Parses command-line arguments for various input options (image, directory, video, etc.).
- Sets up logging for tracking the execution process.
- Resizes images to a target size for consistent processing.
- Applies the detection model to images and calculates metrics.
- Loads ground truth annotations and compares them with detections.
- Computes and logs evaluation metrics such as precision, recall, F1-score, accuracy, and mean Intersection over Union (IoU).
- Handles detection on video streams or webcam input.

Key Features:
- Detection: Uses a pre-trained model to detect persons in images and videos.
- Visualization: Annotates images with bounding boxes and confidence scores.
- Metrics Computation: Calculates precision, recall, F1-score, accuracy, and mean IoU for detected objects.
- Confusion Matrix: Generates and displays a confusion matrix to evaluate detection performance.

Usage:
1. Model Path: Specify the path to the pre-trained model using `-m` or `--model`.
2. Image File: Provide the path to an image file using `-i` or `--image`.
3. Directory of Images: Provide the path to a directory containing images using `-d` or `--directory`.
4. Video File: Provide the path to a video file or use `0` for webcam input using `-v` or `--video`.
5. Detection Threshold: Set the confidence threshold for detection using `-t` or `--threshold`.
6. Ground Truth Directory: Specify the path to a directory containing ground truth annotations using `-gt` or `--groundtruth`.

Requirements:
- argparse: For command-line argument parsing.
- logging: For logging messages.
- os: For file and directory operations.
- json: For handling JSON files.
- re: For regular expression operations.
- detector: Custom module for detection (assumed to be provided).
- PIL (Pillow): For image processing.
- numpy: For numerical operations.
- scikit-learn: For confusion matrix and display.
- matplotlib: For plotting confusion matrices.

Example Command:
```
python Models_Predictions_Experimentals_Dataset.py -m path/to/model -d path/to/directory -t 0.5
```

"""
import argparse
import logging
import os
import json
import re
import detector
from detector import Detector
from PIL import Image
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Define the class file for COCO dataset classes
classFile = "model_data/adas_flir_classes.txt"
TARGET_SIZE = (640, 512)


def arg_parse():
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Person Detection")
    parser.add_argument(
        "-m", "--model",
        help="Model Path. Link to any model can be passed from this URL: ",
        type=str,
    )
    parser.add_argument(
        "-i", "--image",
        help="Path to image file",
        type=str
    )
    parser.add_argument(
        "-d", "--directory",
        help="Path to directory containing images",
        type=str
    )
    parser.add_argument(
        "-v", "--video",
        help="Path to video file, use 0 for webcam",
        type=str
    )
    parser.add_argument(
        "-t", "--threshold",
        help="Threshold for detection confidence",
        type=float,
        default=0.5
    )
    parser.add_argument(
        "-gt", "--groundtruth",
        help="Path to directory containing ground truth annotations",
        type=str
    )
    return parser.parse_args()


def setup_logger():
    """
    Set up the logger.
    """
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    handler = logging.FileHandler(filename='logs/person_detection_logs.log')
    handler.setFormatter(formatter)

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    logger.addHandler(handler)

    return logger


def resize_image(image_path, output_path, target_size):
    """
    Resize an image to the target size.
    """
    image = Image.open(image_path)
    resized_image = image.resize(target_size)
    resized_image.save(output_path)


def correct_boxes(boxes, image_h, image_w):
    for i in range(len(boxes)):
        boxes[i].xmin = int(boxes[i].xmin * image_w)
        boxes[i].xmax = int(boxes[i].xmax * image_w)
        boxes[i].ymin = int(boxes[i].ymin * image_h)
        boxes[i].ymax = int(boxes[i].ymax * image_h)


def get_boxes(boxes, labels, thresh):
    v_boxes, v_labels, v_scores = list(), list(), list()
    # enumerate all boxes
    for box in boxes:
        # enumerate all possible labels
        for i in range(len(labels)):
            # check if the threshold for this label is high enough
            if box.classes > thresh:
                print('box.confidence ', box.confidence)
                v_boxes.append(box)
                v_labels.append(labels[i])
                v_scores.append(box.confidence)
                print("Yes box.classes[i] > thresh", box.classes, v_scores)
            # don't break, many labels may trigger for one box
    return v_boxes, v_labels, v_scores


def process_image_directory(directory_path, detector_, threshold):
    """
    Process all images in a directory for person detection and calculate metrics.
    """
    output_dir = os.path.join(directory_path, 'output')
    os.makedirs(output_dir, exist_ok=True)

    all_pred_boxes, all_gt_boxes, all_pred_labels = [], [], []
    scores, counter = 0, 0

    for filename in sorted(os.listdir(directory_path)):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(directory_path, filename)
            output_path = os.path.join(output_dir, filename)

            # Perform detection
            bboxImage, bboxes = detector_.predictImage(image_path, threshold)
            if bboxes is None:
                logger.error(f"Prediction returned None for image: {image_path}")
                continue

            labels = ["person"]
            pred_boxes, pred_labels, pred_scores = get_boxes(bboxes, labels, threshold)

            if pred_boxes is None or pred_labels is None:
                logger.error(f"Conversion failed for image: {image_path}")
                continue

            all_pred_boxes.extend(pred_boxes)
            all_pred_labels.extend(pred_labels)

            for i in range(len(pred_boxes)):
                print(f"Predicted: {pred_labels[i]}, Score: {pred_scores[i]}")
                scores += pred_scores[i]
                counter += 1

            gt_path = os.path.join(directory_path,
                                   filename.replace('.jpg', '.json').replace('.jpeg', '.json').replace('.png', '.json'))
            if os.path.exists(gt_path):
                gt_boxes = load_ground_truth(gt_path)
                print("**************START****************")
                print(pred_boxes, pred_labels, pred_scores)
                print(gt_boxes)
                print("**************END****************")
                all_gt_boxes.extend(gt_boxes)
            else:
                logger.warning(f"Ground truth file not found: {gt_path}")

            # resize_image(image_path, output_path, TARGET_SIZE)

    print('Prediction score:', (scores / counter) if counter > 0 else 0)
    metrics = calculate_metrics(all_pred_boxes, all_gt_boxes, all_pred_labels, iou_threshold=threshold)
    log_metrics(metrics)

    # Calculate AP for 'person'  pred_boxes, gt_boxes, iou_threshold=0.5
    ap_person = compute_precision_recall_ap_for_person(all_pred_boxes, all_gt_boxes,
                                                       confidence_threshold=0.25, iou_threshold=threshold)
    print(f'Average Precision for "person": {ap_person:.4f}')


def _interval_overlap(interval_a, interval_b):
    x1, x2 = interval_a
    x3, x4 = interval_b
    if x3 < x1:
        if x4 < x1:
            return 0
        else:
            return min(x2, x4) - x1
    else:
        if x2 < x3:
            return 0
        else:
            return min(x2, x4) - x3


def bbox_iou(box1, box2):
    intersect_w = _interval_overlap([box1.xmin, box1.xmax], [box2.xmin, box2.xmax])
    intersect_h = _interval_overlap([box1.ymin, box1.ymax], [box2.ymin, box2.ymax])
    intersect = intersect_w * intersect_h
    w1, h1 = box1.xmax - box1.xmin, box1.ymax - box1.ymin
    w2, h2 = box2.xmax - box2.xmin, box2.ymax - box2.ymin
    union = w1 * h1 + w2 * h2 - intersect
    return float(intersect) / union


def load_ground_truth(gt_path):
    """
    Load ground truth bounding boxes and labels from a JSON file.
    """
    gt_bboxes = []
    with open(gt_path, 'r') as file:
        data = json.load(file)
    for shape in data['shapes']:
        points = shape['points']
        # Convert points to bounding box format [x_min, y_min, x_max, y_max]
        x_min = min(points[0][0], points[1][0])
        y_min = min(points[0][1], points[1][1])
        x_max = max(points[0][0], points[1][0])
        y_max = max(points[0][1], points[1][1])
        box = detector.BoundBox(x_min, y_min, x_max, y_max, 1, 1)
        gt_bboxes.append(box)
    return gt_bboxes


def compute_precision_recall_ap_for_person(pred_boxes, gt_boxes, class_index=1, confidence_threshold=0.25,
                                           iou_threshold=0.5):
    # Filter predictions based on confidence threshold
    filtered_preds = [box for box in pred_boxes if box.confidence >= confidence_threshold]

    # If there are no predictions, return AP = 0
    if len(filtered_preds) == 0:
        return 0

    # Sort predictions by confidence in descending order
    sorted_preds = sorted(filtered_preds, key=lambda x: x.confidence, reverse=True)

    # Total number of ground truth boxes for the specified class index
    ground_truth_person_boxes = [box for box in gt_boxes if box.classes == class_index]
    total_gts = len(ground_truth_person_boxes)

    # If there are no ground truth boxes, return AP = 0
    if total_gts == 0:
        return 0

    # Initialize arrays for true positives and false positives
    tp = np.zeros(len(sorted_preds))
    fp = np.zeros(len(sorted_preds))

    matched_gt = set()  # To keep track of matched ground truth boxes

    for i, pred in enumerate(sorted_preds):
        best_iou = 0
        best_gt_idx = -1

        # Find the ground truth box with the best IoU for the current prediction
        for j, gt in enumerate(ground_truth_person_boxes):
            iou = calculate_iou(pred, gt)
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = j

        # If the IoU is above the threshold and the ground truth box is not already matched
        if best_iou >= iou_threshold and best_gt_idx not in matched_gt:
            tp[i] = 1
            matched_gt.add(best_gt_idx)  # Mark this ground truth box as matched
        else:
            fp[i] = 1

    # Calculate cumulative sums of true positives and false positives
    tp_cumsum = np.cumsum(tp)
    fp_cumsum = np.cumsum(fp)

    # Calculate precision and recall
    precisions = tp_cumsum / (tp_cumsum + fp_cumsum)
    recalls = tp_cumsum / total_gts

    # Add endpoints to precision-recall curve
    precisions = np.concatenate(([0], precisions, [0]))
    recalls = np.concatenate(([0], recalls, [1]))

    # Ensure precision is monotonically decreasing
    precisions = np.maximum.accumulate(precisions[::-1])[::-1]

    # Calculate Average Precision (AP) using the trapezoidal rule
    average_precision = np.trapz(precisions, recalls)

    return average_precision


def calculate_iou(box1, box2):
    # Calculate Intersection over Union (IoU) between two bounding boxes
    x_min = max(box1.xmin, box2.xmin)
    y_min = max(box1.ymin, box2.ymin)
    x_max = min(box1.xmax, box2.xmax)
    y_max = min(box1.ymax, box2.ymax)

    inter_area = max(0, x_max - x_min) * max(0, y_max - y_min)
    box1_area = (box1.xmax - box1.xmin) * (box1.ymax - box1.ymin)
    box2_area = (box2.xmax - box2.xmin) * (box2.ymax - box2.ymin)

    union_area = box1_area + box2_area - inter_area

    return inter_area / union_area if union_area > 0 else 0


def calculate_metrics(pred_boxes, gt_boxes, labels, iou_threshold):
    TP, FP, FN = 0, 0, 0
    class_counts = {label: {"TP": 0, "FP": 0, "FN": 0} for label in labels}
    all_gt_labels, all_pred_labels = [], []

    # Ensure match_boxes correctly matches GT and predictions
    matches = match_boxes(pred_boxes, gt_boxes, iou_threshold)

    # Iterate through matches and update TP, FP, FN
    for gt, pred in matches:
        gt_label = gt.get_label()
        pred_label = pred.get_label()

        all_gt_labels.append(labels[gt_label])
        all_pred_labels.append(labels[pred_label])

        if gt_label == pred_label:
            TP += 1
            class_counts[labels[gt_label]]["TP"] += 1
        else:
            FP += 1
            FN += 1
            class_counts[labels[gt_label]]["FN"] += 1
            class_counts[labels[pred_label]]["FP"] += 1

    # Handle ground truth boxes not matched to any prediction
    for gt in gt_boxes:
        if not any(gt == match[0] for match in matches):
            FN += 1
            all_gt_labels.append(labels[gt.get_label()])
            all_pred_labels.append('background')
            class_counts[labels[gt.get_label()]]["FN"] += 1

    # Handle predicted boxes not matched to any ground truth
    for pred in pred_boxes:
        if not any(pred == match[1] for match in matches):
            FP += 1
            all_gt_labels.append('background')
            all_pred_labels.append(labels[pred.get_label()])
            class_counts[labels[pred.get_label()]]["FP"] += 1

    # Calculate overall metrics
    precision = TP / (TP + FP) if TP + FP > 0 else 0
    recall = TP / (TP + FN) if TP + FN > 0 else 0
    f1_score_ = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0
    accuracy = TP / (TP + FP + FN) if TP + FP + FN > 0 else 0

    # Calculate metrics for each class
    for label in labels:
        tp = class_counts[label]["TP"]
        fp = class_counts[label]["FP"]
        fn = class_counts[label]["FN"]

        class_precision = tp / (tp + fp) if tp + fp > 0 else 0
        class_recall = tp / (tp + fn) if tp + fn > 0 else 0
        class_f1_score = 2 * (class_precision * class_recall) / (
                class_precision + class_recall) if class_precision + class_recall > 0 else 0

        # Uncomment if per-class metrics are needed
        # print(f"Class: {label}")
        # print(f"Precision: {class_precision:.2f}")
        # print(f"Recall: {class_recall:.2f}")
        # print(f"F1-score: {class_f1_score:.2f}")
        # print("---")

    print("Overall Metrics")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1-score: {f1_score_:.2f}")
    print(f"Accuracy: {accuracy:.2f}")

    # Confusion Matrix
    labels = ['person']  # Ensure this matches your class labels
    background = 'background'
    all_labels = labels + [background]

    filtered_gt_labels = [label for label in all_gt_labels if label in all_labels]
    filtered_pred_labels = [label for label in all_pred_labels if label in all_labels]

    conf_matrix = confusion_matrix(filtered_gt_labels, filtered_pred_labels, labels=all_labels)
    cm_display = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=all_labels)
    cm_display.plot(cmap='BuGn')
    plt.title('Confusion Matrix')
    plt.show()

    # # Mean IoU
    # iou_scores = 0
    # for gt, pred in matches:
    #     iou = compute_iou(gt, pred)
    #     if isinstance(iou, (int, float)) and iou >= 0:
    #         iou_scores += iou
    #
    # mean_iou = iou_scores / len(gt_boxes)
    # print(f"Mean IoU: {mean_iou:.2f}")

    #     # Mean IoU
    iou_scores = [compute_iou(gt, pred) for gt, pred in matches if gt.get_label() == pred.get_label()]
    mean_iou = np.mean(iou_scores) if iou_scores else 0
    print(f"Mean IoU: {mean_iou:.2f}")

    iou_scores = [compute_iou(gt, pred) for gt, pred in matches if
                  isinstance(compute_iou(gt, pred), (int, float)) and compute_iou(gt, pred) >= 0]
    mean_iou = np.mean(iou_scores) if iou_scores else 0
    print(f"Mean IoU: {mean_iou:.2f}")

    return precision, recall, f1_score_, accuracy, mean_iou, conf_matrix, class_counts


def match_boxes(pred_boxes, gt_boxes, iou_threshold):
    matches = []
    for gt in gt_boxes:
        best_iou = 0
        best_pred = None
        for pred in pred_boxes:
            iou = bbox_iou(gt, pred)
            if iou > best_iou:
                best_iou = iou
                best_pred = pred
        if best_iou >= iou_threshold:
            matches.append((gt, best_pred))
    return matches


# Utility function to calculate IoU
def compute_iou(box1, box2):
    x1_inter = max(box1.xmin, box2.xmin)
    y1_inter = max(box1.ymin, box2.ymin)
    x2_inter = min(box1.xmax, box2.xmax)
    y2_inter = min(box1.ymax, box2.ymax)

    inter_area = max(0, x2_inter - x1_inter) * max(0, y2_inter - y1_inter)

    box1_area = (box1.xmax - box1.xmin) * (box1.ymax - box1.ymin)
    box2_area = (box2.xmax - box2.xmin) * (box2.ymax - box2.ymin)

    union_area = box1_area + box2_area - inter_area

    return inter_area / union_area


def log_metrics(metrics):
    precision, recall, f1_score_, accuracy, mean_iou, conf_matrix, class_counts = metrics
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1-score: {f1_score_:.2f}")
    print(f"Accuracy: {accuracy:.2f}")
    print(f"Mean IoU: {mean_iou:.2f}")
    print("Confusion Matrix:")
    print(conf_matrix)


def main():
    """
    Main function to execute person detection.
    """
    args = arg_parse()
    global logger
    logger = setup_logger()

    threshold = args.threshold
    model_URL = args.model
    imagePath = args.image
    directoryPath = args.directory
    videoPath = args.video
    GroundTruthDir = directoryPath

    detector_model = Detector()
    detector_model.readClasses(classFile)

    logger.info(f"Downloading model from {model_URL}...")
    detector_model.downloadModel(model_URL)
    logger.info("Model downloaded successfully.")

    logger.info("Loading model...")
    detector_model.loadModel()
    logger.info("Model loaded successfully.")

    if imagePath:
        logger.info(f"Running person detection on image: {imagePath}")
        pred_boxes, pred_labels = detector_model.predictImage(imagePath, threshold)
        resize_image(imagePath, imagePath, TARGET_SIZE)
        if GroundTruthDir:
            base_name = re.sub(r'\.(jpg|jpeg|JPG)$', '.json', os.path.basename(imagePath))
            gt_path = os.path.join(GroundTruthDir, base_name)
            gt_boxes, gt_labels = load_ground_truth(gt_path)
            metrics = calculate_metrics(pred_boxes, gt_boxes, pred_labels, gt_labels)
            log_metrics(metrics)
    elif directoryPath:
        logger.info(f"Running person detection on images in directory: {directoryPath}")
        process_image_directory(directoryPath, detector_model, threshold)
    elif videoPath:
        if videoPath == "0":
            logger.info("Running person detection on webcam...")
            detector_model.predictVideo(0, threshold)
        else:
            logger.info(f"Running person detection on video: {videoPath}")
            detector_model.predictVideo(videoPath, threshold)
    else:
        print("Please specify an image, directory, or video path, or 0 for webcam. Use -h for help")
        logger.error("No image, directory, or video path specified.")


if __name__ == "__main__":
    main()
