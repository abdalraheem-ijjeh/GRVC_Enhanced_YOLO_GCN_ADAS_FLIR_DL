"""
Author: Abdalraheem A. Ijjeh
Affiliation: Postdoc researcher / GRVC Robotics Laboratory, University of Seville, 41092 Seville, Spain

Description:
This script processes images using an object detection model, calculates various performance metrics, and generates visualizations and outputs. It is designed to work with object detection models and requires various libraries for processing, visualization, and metric calculation.

The script performs the following tasks:
- Loads an object detection model from a specified file path.
- Processes images and performs predictions using the loaded model.
- Calculates precision, recall, F1-score, accuracy, mean Intersection over Union (IoU), and confusion matrix.
- Draws bounding boxes on images and saves them with predictions.
- Computes Average Precision (AP) for specified classes.
- Logs and visualizes metric results and predictions.

Key Features:
- Metrics Calculation: Computes precision, recall, F1-score, accuracy, mean IoU, and confusion matrix for the model predictions.
- Bounding Box Visualization: Draws and saves images with bounding boxes around detected objects.
- AP Computation: Calculates Average Precision (AP) for specific classes based on precision-recall metrics.
- Performance Logging: Logs metric results and outputs visualizations for analysis.

Requirements:
- TensorFlow: For loading and using the object detection model.
- OpenCV (cv2): For image processing and visualization.
- NumPy: For numerical operations.
- Matplotlib & Seaborn: For plotting and visualizing results.
- Sklearn: For computing metrics and confusion matrices.
- JSON: For reading and handling data files.
- OS: For file and directory operations.

Usage:
1. Model Path: Specify the path to the object detection model file using `model_path`.
2. Image Processing: Use the `predict_image()` function to process images and make predictions.
3. Metrics Calculation: Metrics such as precision, recall, F1-score, and accuracy are automatically calculated.
4. Bounding Box Drawing: Bounding boxes are drawn on images and saved along with the predictions.

Example Command:
```python
python Predicting_DL_detection_model_experimental_dataset.py
```

Notes:
- Ensure that all required libraries are installed before running the script.
- The script assumes that the model file is in a compatible format and can be loaded directly using TensorFlow/Keras.
- Adjust paths and parameters according to your dataset and model requirements.

"""

import json
import os
import time

import keras
import sklearn
import tensorflow as tf
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from numpy import expand_dims
import seaborn as sns
import numpy as np
import cv2
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, average_precision_score


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
    # Mean IoU
    iou_scores = [bbox_iou(gt, pred) for gt, pred in matches if gt.get_label() == pred.get_label()]
    mean_iou = np.mean(iou_scores) if iou_scores else 0
    print(f"Mean IoU: {mean_iou:.2f}")
    return precision, recall, f1_score_, accuracy, mean_iou, conf_matrix, class_counts


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


def collect_data(pred_boxes, gt_boxes, labels, iou_threshold=0.5):
    all_gt_labels = []
    all_pred_scores = []
    all_pred_labels = []

    matches = match_boxes(pred_boxes, gt_boxes, iou_threshold)

    for gt, pred in matches:
        gt_label = labels[gt.get_label()]
        pred_label = labels[pred.get_label()]
        pred_score = pred.get_score()  # Assumes prediction score is available

        all_gt_labels.append(gt_label)
        all_pred_labels.append(pred_label)
        all_pred_scores.append(pred_score)

    # Handle unmatched ground truths and predictions
    for gt in gt_boxes:
        if not any(gt == match[0] for match in matches):
            all_gt_labels.append(labels[gt.get_label()])
            all_pred_labels.append('background')  # No matching prediction
            all_pred_scores.append(0)  # Score of 0 for unmatched ground truths

    for pred in pred_boxes:
        if not any(pred == match[1] for match in matches):
            all_gt_labels.append('background')
            all_pred_labels.append(labels[pred.get_label()])
            all_pred_scores.append(pred.get_score())

    return all_gt_labels, all_pred_labels, all_pred_scores


def compute_precision_recall_ap_for_person(pred_boxes, gt_boxes, class_index, confidence_threshold, iou_threshold):
    # Filter predictions based on confidence threshold
    filtered_preds = [box for box in pred_boxes if box.confidence >= confidence_threshold]

    # Sort predictions by confidence in descending order
    sorted_preds = sorted(filtered_preds, key=lambda x: x.confidence, reverse=True)

    # Total number of ground truth boxes for the specified class index
    ground_truth_person_boxes = [box for box in gt_boxes if box.classes == class_index]
    total_gts = len(ground_truth_person_boxes)
    # Debugging: Print the number of predictions and ground truth boxes
    print(f"Number of predictions after filtering: {len(sorted_preds)}")
    print(f"Number of ground truth boxes for class index {class_index}: {total_gts}")

    # Initialize arrays for true positives and false positives
    tp = np.zeros(len(sorted_preds))
    fp = np.zeros(len(sorted_preds))

    matched_gt = set()  # To keep track of matched ground truth boxes

    for i, pred in enumerate(sorted_preds):
        best_iou = 0
        best_gt_idx = -1

        for j, gt in enumerate(ground_truth_person_boxes):
            iou = calculate_iou(pred, gt)
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = j

        if best_iou >= iou_threshold and best_gt_idx not in matched_gt:
            tp[i] = 1
            matched_gt.add(best_gt_idx)
        else:
            fp[i] = 1

    # Calculate cumulative sums
    tp_cumsum = np.cumsum(tp)
    fp_cumsum = np.cumsum(fp)

    # Calculate precision and recall
    precisions = tp_cumsum / (tp_cumsum + fp_cumsum)
    recalls = tp_cumsum / total_gts if total_gts > 0 else np.zeros(len(tp_cumsum))

    # Add endpoints to precision-recall curve
    precisions = np.concatenate(([0], precisions, [0]))
    recalls = np.concatenate(([0], recalls, [1]))

    # Ensure precision is monotonically decreasing
    precisions = np.maximum.accumulate(precisions[::-1])[::-1]

    # Calculate Average Precision using trapezoidal rule
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


#############################################################################

def _sigmoid(x):
    return 1. / (1. + np.exp(-x))


class BoundBox:
    def __init__(self, xmin, ymin, xmax, ymax, confidence=None, classes=None):
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax
        self.confidence = confidence
        self.classes = classes
        self.label = -1
        self.score = -1

    def get_label(self):
        if self.label == -1:
            self.label = np.argmax(self.classes)

        return self.label

    def get_score(self):
        if self.score == -1:
            self.score = self.classes[self.get_label()]

        return self.score


def decode_netout(netout, anchors, obj_thresh, net_h, net_w):
    boxes = []
    grid_h, grid_w = netout.shape[:2]
    nb_box = 3
    netout = netout.reshape((grid_h, grid_w, nb_box, -1))

    nb_class = netout.shape[-1] - 5
    netout[..., :2] = _sigmoid(netout[..., :2])
    netout[..., 4:] = _sigmoid(netout[..., 4:])
    netout[..., 5:] = netout[..., 4][..., np.newaxis] * netout[..., 5:]
    netout[..., 5:] *= netout[..., 5:] > obj_thresh

    for row in range(grid_h):
        for col in range(grid_w):
            for b in range(nb_box):
                objectness = netout[row, col, b, 4]
                if objectness <= obj_thresh:
                    continue
                x, y, w, h = netout[row, col, b, :4]
                x = (col + x) / grid_w
                y = (row + y) / grid_h
                w = anchors[2 * b] * np.exp(w) / net_w
                h = anchors[2 * b + 1] * np.exp(h) / net_h
                classes = netout[row, col, b, 5:]
                box = BoundBox(x - w / 2, y - h / 2, x + w / 2, y + h / 2, objectness, classes)
                boxes.append(box)
    return boxes


def draw_boxes(filename, v_boxes_, v_labels_, v_scores_):
    image_ = cv2.imread(filename)
    # Plot each box
    shapes = []
    if not os.path.exists(path + "_YOLO_GCN"):
        os.makedirs(path + "_YOLO_GCN")
    for i in range(len(v_boxes_)):
        box = v_boxes_[i]
        # Get coordinates
        y1, x1, y2, x2 = box.ymin, box.xmin, box.ymax, box.xmax
        # Calculate the width and height of the box
        width, height = 640, 512
        # Draw the rectangle around the detected object
        cv2.rectangle(image_, (x1, y1), (x2, y2), (0, 255, 0), 2)
        # Draw text and score in the top left corner
        label = f"{v_labels_[i]} ({v_scores_[i]:.3f})"
        cv2.putText(image_, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
        #############################################################
        # Store predictions in JSON file per model and input set
        #############################################################
        points = [
            [x1, y1],
            [x2, y2]
        ]

        shape = {
            "label": "person",
            "points": points,
            "group_id": None,
            "description": "",
            "confidence": v_scores_[i],
            "shape_type": "rectangle",
            "flags": {},
            "mask": None
        }
        shapes.append(shape)
        json_data = {
            "version": "5.4.1",
            "flags": {},
            "shapes": shapes,
            "imagePath": os.path.basename(filename),
            "imageData": None,
            "imageHeight": height,
            "imageWidth": width
        }

        # Save to JSON file
        json_filename = os.path.splitext(filename)[0] + "_" + "YOLO_GCN" + '_Pred_.json'
        with open(os.path.join(path + "_YOLO_GCN", json_filename), 'w') as json_file:
            json.dump(json_data, json_file, indent=4)
        #############################################################

    # Show the image
    # cv2.imshow(filename, image_)

    cv2.imwrite(os.path.join((path + "_YOLO_GCN"), 'Pred_%s_%s_' % (threshold, occlusion_percentage) + filename),
                image_)
    # cv2.waitKey()
    # cv2.destroyAllWindows()


def correct_yolo_boxes(boxes, image_h, image_w, net_h, net_w):
    new_w, new_h = net_w, net_h
    for i in range(len(boxes)):
        x_offset, x_scale = (net_w - new_w) / 2. / net_w, float(new_w) / net_w
        y_offset, y_scale = (net_h - new_h) / 2. / net_h, float(new_h) / net_h
        boxes[i].xmin = int((boxes[i].xmin - x_offset) / x_scale * image_w)
        boxes[i].xmax = int((boxes[i].xmax - x_offset) / x_scale * image_w)
        boxes[i].ymin = int((boxes[i].ymin - y_offset) / y_scale * image_h)
        boxes[i].ymax = int((boxes[i].ymax - y_offset) / y_scale * image_h)


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


def do_nms(boxes_, nms_thresh):
    if len(boxes_) > 0:
        nb_class = len(boxes_[0].classes)
    else:
        return
    for c in range(nb_class):
        sorted_indices = np.argsort([-box.classes[c] for box in boxes_])
        for i in range(len(sorted_indices)):
            index_i = sorted_indices[i]
            if boxes_[index_i].classes[c] == 0: continue
            for j in range(i + 1, len(sorted_indices)):
                index_j = sorted_indices[j]
                if bbox_iou(boxes_[index_i], boxes_[index_j]) >= nms_thresh:
                    boxes_[index_j].classes[c] = 0


def get_boxes(boxes, labels, thresh):
    v_boxes, v_labels, v_scores = list(), list(), list()
    # enumerate all boxes
    for box in boxes:
        # enumerate all possible labels
        for i in range(len(labels)):
            # check if the threshold for this label is high enough
            if box.classes[i] > thresh:
                v_boxes.append(box)
                v_labels.append(labels[i])
                v_scores.append(box.classes[i] * 100)
            # don't break, many labels may trigger for one box
    return v_boxes, v_labels, v_scores


# def load_image_pixels(filename, shape):
#     # load the image to get its shape
#     image = load_img(filename)
#     width, height = image.size
#     # load the image with the required size
#     image = load_img(filename, target_size=shape)
#     # convert to numpy array
#     image = img_to_array(image)
#     # scale pixel values to [0, 1]
#     image = image.astype('float32')
#     image /= 255.0
#     # add a dimension so that we have one sample
#     image = expand_dims(image, 0)
#     return image, width, height
def load_image_pixels(filename, shape):
    # load the image to get its shape

    img = cv2.imread(filename)
    image = img.copy()
    # image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = np.expand_dims(image, axis=-1)
    image = cv2.merge((image, image, image))
    height, width, depth = image.shape
    #########################################################
    # blurred = cv2.GaussianBlur(image, (7, 7), 0)
    # image = cv2.addWeighted(image, 1.5, blurred, -2.0, 0)
    # ########################################################
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)
    l_channel, a, b = cv2.split(lab)
    l_channel = cv2.normalize(l_channel, None, 0, 255, cv2.NORM_MINMAX)
    clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(4, 4))
    cl = clahe.apply(l_channel)
    limg = cv2.merge((cl, a, b))
    image = cv2.cvtColor(limg, cv2.COLOR_Lab2RGB)
    #########################################################
    image = cv2.resize(image, (416, 416))
    # load the image with the required size
    # image = load_img(filename, target_size=shape)
    # convert to a numpy array
    image = img_to_array(image)
    # scale pixel values to [0, 1]
    image = image.astype('float32')

    image /= 255.0
    # add a dimension so that we have one sample
    image = expand_dims(image, 0)
    return image, width, height, img


def read_bboxes(gt_file):
    print(gt_file)
    with open(gt_file, 'r') as f:
        data = json.load(f)
    shapes = data['shapes']
    bounding_boxes = []

    for shape in shapes:
        if shape['shape_type'] == 'rectangle':
            points = shape['points']
            x_min = min(points[0][0], points[1][0])
            y_min = min(points[0][1], points[1][1])
            x_max = max(points[0][0], points[1][0])
            y_max = max(points[0][1], points[1][1])
            box = BoundBox(x_min, y_min, x_max, y_max, 1, 1)
            bounding_boxes.append(box)
    print(bounding_boxes)
    return bounding_boxes


def predict_image(path_):
    os.chdir(path_)
    pred_boxes_list = []  # List to hold predicted boxes for all images
    pred_labels_list = []  # List to hold predicted labels for all images
    gt_boxes_list = []  # List to hold ground truth boxes for all images
    scores = 0
    counter = 0
    t = 0
    f_counter = 0
    for photo_filename in sorted(os.listdir()):
        if photo_filename.endswith(('T.JPG', 'T.jpg')):
            f_counter += 1
            gt_file = photo_filename.replace('T.jpg', 'T.json')
            gt_bboxes = read_bboxes(gt_file)
            gt_boxes_list.append(gt_bboxes)

            anchors = [[116, 90, 156, 198, 373, 326], [30, 61, 62, 45, 59, 119], [10, 13, 16, 30, 33, 23]]
            class_threshold = threshold
            input_w, input_h = 416, 416

            try:
                image, image_w, image_h, img = load_image_pixels(photo_filename, (input_w, input_h))
                t1 = time.time()
                yhat = model.predict(image)
                t2 = time.time()
                t += (t2 - t1)
                # Decode the predictions
                boxes = [decode_netout(yhat[i][0], anchors[i], class_threshold, input_h, input_w) for i in
                         range(len(yhat))]

                # Flatten the list of boxes
                boxes = [box for sublist in boxes for box in sublist]

                # Correct bounding box sizes
                correct_yolo_boxes(boxes, image_h, image_w, input_h, input_w)

                # Non-max suppression
                do_nms(boxes, threshold)

                labels = ["person"]
                pred_boxes, pred_labels, pred_scores = get_boxes(boxes, labels, class_threshold)
                # if len(pred_boxes) > 0:
                #     valid_imgs.append(photo_filename)
                draw_boxes(filename=photo_filename, v_boxes_=pred_boxes, v_labels_=pred_labels, v_scores_=pred_scores)
                pred_boxes_list.append(pred_boxes)
                pred_labels_list.append(pred_labels)

                for i in range(len(pred_boxes)):
                    print(f"Predicted: {pred_labels[i]}, Score: {pred_scores[i]}")
                    scores += pred_scores[i]
                    counter += 1

            except Exception as e:
                print(f"Error processing file {photo_filename}: {e}")

    print('prediction score', (scores / counter))
    # Flatten pred_labels_list and gt_boxes_list for metric calculation
    flat_pred_boxes = [box for sublist in pred_boxes_list for box in sublist]
    flat_pred_labels = [label for sublist in pred_labels_list for label in sublist]
    flat_gt_boxes = [box for sublist in gt_boxes_list for box in sublist]

    # Calculate metrics
    metrics = calculate_metrics(flat_pred_boxes, flat_gt_boxes, flat_pred_labels, iou_threshold=threshold)
    log_metrics(metrics)

    # Calculate AP for 'person'
    ap_person = compute_precision_recall_ap_for_person(flat_pred_boxes, flat_gt_boxes, class_index=1,
                                                       confidence_threshold=0.25, iou_threshold=threshold)
    print(f'Average Precision for "person": {ap_person:.4f}')
    fsp = 1 / (t / f_counter)
    print(f'FPS: {fsp:.4f}')


def log_metrics(metrics):
    precision, recall, f1_score_, accuracy, mean_iou, conf_matrix, class_counts = metrics
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1-score: {f1_score_:.2f}")
    print(f"Accuracy: {accuracy:.2f}")
    print(f"Mean IoU: {mean_iou:.2f}")
    print("Confusion Matrix:")
    print(conf_matrix)


if __name__ == "__main__":
    valid_imgs = []
    count = 0
    model = tf.keras.models.load_model('Yolo_GCN_model.keras', compile=False)
    model.summary()
    dataset_path = "/home/abdalraheem/Documents/GitHub/TEMA_Visual_IR_DL_Detection/datasets/Smoke_dataset/"

    occlusion_percentage = 0.5
    path = (
        os.path.join(dataset_path, f"DJI_202406032323_029/valid_thermal_images/occluded_images_{occlusion_percentage}"))

    # DAY 1: DJI_202407230824_030/occluded_images_0.3
    # DAY 2: DJI_202408011319_031/valid_thermal_images/occluded_images_0.3
    # DAY 3: DJI_202406032323_029/valid_thermal_images/occluded_images_0.3

    # trial = 'DJI_202407230824_030/'
    # folder = 'occluded_images/'
    # path = ("/home/abdalraheem/Documents/GitHub/TEMA_Visual_IR_DL_Detection/datasets/Smoke_dataset"
    #         "/DJI_202407230824_030/valid_thermal_images")  # os.path.join(dataset_path, trial, folder)
    threshold = 0.40
    predict_image(path)
    # with open('valid_file.txt', 'w') as f:
    #     for line in valid_imgs:
    #         f.write(f"{line}\n")
