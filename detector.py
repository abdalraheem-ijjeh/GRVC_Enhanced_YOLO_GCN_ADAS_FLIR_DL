import json
import logging
import cv2
import numpy as np
import os
import tensorflow as tf
import time
from tensorflow.python.keras.utils.data_utils import get_file

np.random.seed(123)


class BoundBox:
    def __init__(self, xmin, ymin, xmax, ymax, confidence, classes):
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


class Detector:
    def __init__(self) -> None:
        pass

    def readClasses(self, classesFilePath):
        with open(classesFilePath, "r") as f:
            self.classesList = [line.rstrip() for line in f]

        # colors for drawing bounding boxes
        self.colorsList = np.random.uniform(0, 255, size=(len(self.classesList), 3))

    def downloadModel(self, modelURL):
        fileName = os.path.basename(modelURL)
        self.modelName = fileName[:fileName.index(".")]

        self.cacheDir = "./pretrained_models/"
        os.makedirs(self.cacheDir, exist_ok=True)
        get_file(fileName, modelURL, cache_dir=self.cacheDir, cache_subdir="checkpoints", extract=True)

    def loadModel(self):
        print("Loading model..." + self.modelName)
        tf.keras.backend.clear_session()
        self.model = tf.saved_model.load(os.path.join(self.cacheDir, "checkpoints", self.modelName, "saved_model"))
        print("Model " + self.modelName + " loaded successfully!")

    def createBoundingBox(self, image, imagePath, threshold=0.5):
        inputTensor = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2RGB)
        inputTensor = tf.convert_to_tensor(inputTensor, dtype=tf.uint8)
        inputTensor = inputTensor[tf.newaxis, ...]
        detections = self.model(inputTensor)
        bboxes = detections["detection_boxes"][0].numpy()
        classIndexes = detections["detection_classes"][0].numpy().astype(np.int32)
        classScores = detections["detection_scores"][0].numpy()

        imH, imW, imC = image.shape
        bboxIdx = tf.image.non_max_suppression(bboxes, classScores, max_output_size=100, iou_threshold=threshold,
                                               score_threshold=threshold)
        print("(bboxIdx: ", bboxIdx)
        pred_bboxes = []

        if len(bboxIdx) != 0:
            shapes = []
            for i in range(len(bboxIdx)):

                bbox = tuple(bboxes[i].tolist())
                classConfidence = (classScores[i] * 100)
                classIndex = classIndexes[i]

                if classIndex == 1:
                    print("bbox, classConfidence, classIndex ", bbox, classConfidence, classIndex)
                    classLabelText = self.classesList[classIndex].upper()
                    classColor = self.colorsList[classIndex]

                    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s",
                                                  datefmt="%d-%m-%Y %H:%M:%S")
                    handler = logging.FileHandler(filename='logs/person_detection_logs.log')
                    handler.setFormatter(formatter)
                    logger = logging.getLogger()
                    logger.setLevel(logging.DEBUG)
                    logger.addHandler(handler)

                    displayText = '{} {:.2f}%'.format(classLabelText, classConfidence)
                    ymin, xmin, ymax, xmax = bbox
                    xmin, xmax, ymin, ymax = (xmin * imW, xmax * imW, ymin * imH, ymax * imH)
                    xmin, xmax, ymin, ymax = int(xmin), int(xmax), int(ymin), int(ymax)

                    cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
                    # cv2.putText(image, displayText, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 255, 0), 1,
                    #             cv2.LINE_AA)
                    cv2.putText(image, 'Person', (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2,
                                cv2.LINE_AA)

                    logger.info(displayText)
                    bbox = BoundBox(xmin, ymin, xmax, ymax, classConfidence, classIndex)
                    pred_bboxes.append(bbox)

                    #############################################################
                    # Store predictions in JSON file per model and input set
                    #############################################################
                    points = [
                        [xmin, ymin],
                        [xmax, ymax]
                    ]

                    shape = {
                        "label": classLabelText.lower(),
                        "points": points,
                        "group_id": None,
                        "description": "",
                        "confidence": classConfidence,
                        "shape_type": "rectangle",
                        "flags": {},
                        "mask": None
                    }
                    shapes.append(shape)
                    json_data = {
                        "version": "5.4.1",
                        "flags": {},
                        "shapes": shapes,
                        "imagePath": os.path.basename(imagePath),
                        "imageData": None,
                        "imageHeight": imH,
                        "imageWidth": imW
                    }

                    # Save to JSON file
                    json_filename = os.path.splitext(imagePath)[0] + "_" + self.modelName + '_Pred_.json'
                    with open(json_filename, 'w') as json_file:
                        json.dump(json_data, json_file, indent=4)
                    #############################################################

        return image, pred_bboxes

    def predictImage(self, imagePath, threshold):
        print("imagePath", imagePath)
        image = cv2.imread(imagePath)
        print("image.shape", image.shape)
        start_time = time.time()
        bboxImage, bboxes = self.createBoundingBox(image, imagePath, threshold)
        end_time = time.time()
        processing_time = end_time - start_time
        fps = 1 / processing_time if processing_time > 0 else float('inf')
        print(f"Estimated FPS: {fps:.2f}")
        # cv2.imwrite(self.modelName + ".jpg", bboxImage)
        # cv2.imshow("Result", image)
        # cv2.waitKey()
        # cv2.destroyAllWindows()
        return bboxImage, bboxes

    def predictVideo(self, videoPath, threshold=0.5):
        cap = cv2.VideoCapture(videoPath)
        if not cap.isOpened():
            print("Error opening video stream or file")
            return

        (success, image) = cap.read()
        startTime = 0
        while success:
            currentTime = time.time()

            fps = 1 / (currentTime - startTime)
            startTime = currentTime

            bboxImage = self.createBoundingBox(image, threshold)
            cv2.putText(bboxImage, "FPS: " + str(int(fps)), (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)
            cv2.imshow("Result", bboxImage)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            (success, image) = cap.read()
        cv2.destroyAllWindows()
