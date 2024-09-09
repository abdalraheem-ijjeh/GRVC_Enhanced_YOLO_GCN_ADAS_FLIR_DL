"""
Author: Abdalraheem A. Ijjeh
Affiliation: Postdoc researcher / GRVC Robotics Laboratory, University of Seville, 41092 Seville, Spain

Description:
This script performs person detection using a pre-trained model.
 It supports detection on images, directories of images, and video files.
 The script processes images, calculates detection metrics, and generates visual output such as annotated images
 and confusion matrices. Key functionalities include detection confidence filtering, bounding box annotation,
 and evaluation metrics computation.

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
python Predicting_TELEDYNE_ADAS_FLIR.py -m path/to/model -t path/to/tfrecords -t 0.5
```

"""

import math

import cv2
import numpy as np
import tensorflow as tf
from numpy import expand_dims
from tensorflow.keras.models import load_model
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
import os
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt


def calculate_metrics(pred_boxes, gt_boxes, labels, iou_threshold):
    TP, FP, FN = 0, 0, 0
    class_counts = {label: {"TP": 0, "FP": 0, "FN": 0} for label in labels}
    all_gt_labels, all_pred_labels = [], []
    matches = match_boxes(pred_boxes, gt_boxes, iou_threshold)

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

    for gt in gt_boxes:
        if not any(gt == match[0] for match in matches):
            FN += 1
            all_gt_labels.append(labels[gt.get_label()])
            all_pred_labels.append('background')
            class_counts[labels[gt.get_label()]]["FN"] += 1

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
    labels = ['person']  # List of class labels
    background = 'background'
    all_labels = labels + [background]

    filtered_gt_labels = [label for label in all_gt_labels if label in all_labels]
    filtered_pred_labels = [label for label in all_pred_labels if label in all_labels]

    conf_matrix = confusion_matrix(filtered_gt_labels, filtered_pred_labels, labels=all_labels)
    cm_display = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=all_labels)
    cm_display.plot(cmap='BuGn')
    plt.title('Confusion Matrix')
    plt.show()

    # Mean IoU
    iou_scores = [compute_iou(gt, pred) for gt, pred in matches if gt.get_label() == pred.get_label()]
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


def draw_boxes(original_img, v_boxes_, v_labels_, v_scores_, fname):
    # Load the image
    image_ = original_img
    # print(filename)
    # Convert the image from BGR to RGB (since OpenCV uses BGR by default)
    # image_ = cv2.cvtColor(image_, cv2.COLOR_BGR2RGB)
    # image_ = cv2.cvtColor(image_, cv2.COLOR_RGBA2GRAY)
    # image_ = np.expand_dims(image_, axis=-1)
    # image = np.concatenate([image, image, image], axis=-1)
    # Plot each box
    for i in range(len(v_boxes_)):
        if v_labels_[i] == 'person':
            box = v_boxes_[i]
            # Get coordinates
            y1, x1, y2, x2 = box.ymin, box.xmin, box.ymax, box.xmax
            print(x1, y1, x2, y2)
            # Calculate width and height of the box
            width, height = x2 - x1, y2 - y1
            # Draw the rectangle around the detected object
            cv2.rectangle(image_, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # Draw text and score in top left corner
            label = f"{v_labels_[i]}"  #  ({v_scores_[i]:.3f})
            cv2.putText(image_, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 255, 0), 1, cv2.LINE_AA)

    # Convert the image from RGB back to BGR for display with OpenCV

    # Show the image
    cv2.imshow(fname, image_)
    cv2.waitKey()
    cv2.destroyAllWindows()


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


def load_image_pixels(filename, shape):
    # load the image to get its shape

    img = cv2.imread(filename)
    image = img.copy()
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = np.expand_dims(image, axis=-1)
    image = np.concatenate([image, image, image], axis=-1)
    # image = cv2.applyColorMap(image, cv2.COLORMAP_JET)
    height, width, depth = image.shape
    image = cv2.resize(image, (416, 416))
    # load the image with the required size
    # image = load_img(filename, target_size=shape)
    # convert to numpy array
    image = img_to_array(image)
    # scale pixel values to [0, 1]
    image = image.astype('float32')
    image /= 255.0
    # add a dimension so that we have one sample
    image = expand_dims(image, 0)
    return image, width, height, img


def _parse_function(proto):
    return tf.io.parse_single_example(proto, feature_description)


def load_tfrecord_dataset(tfrecord_file, batch_size):
    raw_dataset = tf.data.TFRecordDataset(tfrecord_file)
    parsed_dataset = raw_dataset.map(_parse_function)
    # dataset = parsed_dataset.batch(batch_size)
    return parsed_dataset


def _decode_image_and_labels(parsed_record):
    filename_ = tf.strings.as_string(parsed_record['filename'])
    image_ = tf.image.decode_jpeg(parsed_record['image_raw'], channels=3)
    height = tf.cast(parsed_record['height'], tf.float32)
    width = tf.cast(parsed_record['width'], tf.float32)

    # Retrieve normalized bounding box coordinates
    xmins = tf.sparse.to_dense(parsed_record['x_mins'])
    xmaxs = tf.sparse.to_dense(parsed_record['x_maxes'])
    ymins = tf.sparse.to_dense(parsed_record['y_mins'])
    ymaxs = tf.sparse.to_dense(parsed_record['y_maxes'])

    # Convert normalized coordinates to pixel values
    xmin = xmins  # * width
    xmax = xmaxs  # * width
    ymin = ymins  # * height
    ymax = ymaxs  # * height

    classes_ID = tf.sparse.to_dense(parsed_record['classes_id'])
    classes_Name = tf.sparse.to_dense(parsed_record['classes_name'])

    return image_, xmin, xmax, ymin, ymax, classes_ID, height, width, filename_


def draw_bboxes_with_labels(image_, x_min, x_max, y_min, y_max, labels_, h_, w_, filename_):
    # Convert the image_ to a format suitable for OpenCV (BGR)
    # image_ = cv2.cvtColor(image_.numpy(), cv2.COLOR_RGB2BGR)
    ######################################################
    image_ = cv2.cvtColor(image_.numpy(), cv2.COLOR_BGR2GRAY)
    image_ = np.expand_dims(image_, axis=-1)
    image_ = np.concatenate([image_, image_, image_], axis=-1)
    # height, width, depth = image_.shape
    # #########################################################
    # # blurred = cv2.GaussianBlur(image_, (5, 5), 0)
    # # image_ = cv2.addWeighted(image_, 1.5, blurred, -0.5, 0)
    # #########################################################
    # lab = cv2.cvtColor(image_, cv2.COLOR_BGR2Lab)
    # l_channel, a, b = cv2.split(lab)
    # # l_channel = cv2.normalize(l_channel, None, 0, 255, cv2.NORM_MINMAX)
    #
    # clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
    # cl = clahe.apply(l_channel)
    # limg = cv2.merge((cl, a, b))
    # image_ = cv2.cvtColor(limg, cv2.COLOR_Lab2RGB)
    ######################################################
    labels_ = tf.strings.as_string(labels_)
    bounding_boxes = []
    for i in range(labels_.shape[0]):
        label = str(labels_[i].numpy().decode('utf-8'))
        if label == '1':
            xmin_i = int(x_min[i] * w_)
            xmax_i = int(x_max[i] * w_)
            ymin_i = int(y_min[i] * h_)
            ymax_i = int(y_max[i] * h_)

            # cv2.rectangle(image_, (xmin_i, ymin_i), (xmax_i, ymax_i), (255, 255, 255), 2)
            box = BoundBox(xmin_i, ymin_i, xmax_i, ymax_i, 1, 1)
            bounding_boxes.append(box)

            # Put label text
            # cv2.putText(image_, label, (xmin_i, ymin_i - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    gt_boxes_list.append(bounding_boxes)
    input_w, input_h = 416, 416
    height, width, depth = image_.shape
    original_image = image_
    image_ = cv2.resize(image_, (416, 416))
    # convert to numpy array
    image_ = img_to_array(image_)
    # scale pixel values to [0, 1]
    image_ = image_.astype('float32')
    image_ /= 255.0
    # add a dimension so that we have one sample
    image_ = np.expand_dims(image_, axis=0)
    yhat = model.predict(image_)
    anchors = [[116, 90, 156, 198, 373, 326], [30, 61, 62, 45, 59, 119], [10, 13, 16, 30, 33, 23]]
    # define the probability threshold for detected objects
    class_threshold = threshold
    boxes = list()
    input_w, input_h = 416, 416

    for i in range(len(yhat)):
        # decode the output of the network
        boxes += decode_netout(yhat[i][0], anchors[i], class_threshold, input_h, input_w)
    # correct the sizes of the bounding boxes for the shape of the image_
    correct_yolo_boxes(boxes, height, width, input_h, input_w)
    # suppress non-maximal boxes
    do_nms(boxes, threshold)
    # define the labels
    labels_ = ["person"]
    # get the details of the detected objects
    pred_boxes, pred_labels, pred_scores = get_boxes(boxes, labels_, class_threshold)
    pred_boxes_list.append(pred_boxes)
    pred_labels_list.append(pred_labels)
    for i in range(len(pred_boxes)):
        print(f"Predicted: {pred_labels[i]}, Score: {pred_scores[i]}")

    # summarize what we found
    draw_boxes(original_image, pred_boxes, pred_labels, pred_scores, filename_)


def predict_image():
    for image, xmin, xmax, ymin, ymax, labels, h, w, filename in decoded_images.take(500):
        print(filename.numpy().decode('utf-8'))
        draw_bboxes_with_labels(image, xmin, xmax, ymin, ymax, labels.numpy(), h, w, filename.numpy().decode('utf-8'))

    # Flatten pred_labels_list and gt_boxes_list for metric calculation
    flat_pred_boxes = [box for sublist in pred_boxes_list for box in sublist]
    flat_pred_labels = [label for sublist in pred_labels_list for label in sublist]
    flat_gt_boxes = [box for sublist in gt_boxes_list for box in sublist]

    # Calculate metrics
    # precision, recall, f1_score, accuracy, mean_iou, conf_matrix = calculate_metrics(flat_pred_boxes, flat_gt_boxes,
    #                                                                                  flat_pred_labels,
    #                                                                                  iou_threshold=0.25)
    # # Calculate metrics
    metrics = calculate_metrics(flat_pred_boxes, flat_gt_boxes, flat_pred_labels, iou_threshold=threshold)
    log_metrics(metrics)

    # Calculate AP for 'person'
    ap_person = compute_precision_recall_ap_for_person(flat_pred_boxes, flat_gt_boxes, class_index=1,
                                                       confidence_threshold=0.25, iou_threshold=0.0)
    print(f'Average Precision for "person": {ap_person:.4f}')


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
    threshold = 0.5
    pred_boxes_list = []  # List to hold predicted boxes for all images
    pred_labels_list = []  # List to hold predicted labels for all images
    gt_boxes_list = []  # List to hold ground truth boxes for all images
    # Define your feature description here based on your dataset
    feature_description = {
        'filename': tf.io.FixedLenFeature([], tf.string),
        'height': tf.io.FixedLenFeature([], tf.int64),
        'width': tf.io.FixedLenFeature([], tf.int64),
        'classes_id': tf.io.VarLenFeature(tf.int64),
        'classes_name': tf.io.VarLenFeature(tf.string),
        'x_mins': tf.io.VarLenFeature(tf.float32),
        'y_mins': tf.io.VarLenFeature(tf.float32),
        'x_maxes': tf.io.VarLenFeature(tf.float32),
        'y_maxes': tf.io.VarLenFeature(tf.float32),
        'image_raw': tf.io.FixedLenFeature([], tf.string),
    }
    valid_imgs = []

    tfrecord_file = 'dataset/video_thermal_test_single_class_person.tfrecord'
    batch_size = 1
    test_dataset = load_tfrecord_dataset(tfrecord_file, batch_size)
    decoded_images = test_dataset.map(_decode_image_and_labels)
    model = tf.keras.models.load_model('Yolo_GCN_model.keras', compile=False)
    model.summary()

    predict_image()
