"""
Author: Abdalraheem A. Ijjeh
Affiliation: Postdoc researcher / GRVC Robotics Laboratory, University of Seville, 41092 Seville, Spain

Description:
    This script performs various image processing tasks, including resizing, rotating, and aligning images.
    It also processes JSON annotation files to visualize and annotate images with bounding boxes.

    The script performs the following tasks:
    - Fits images to a specified width and height while maintaining an aspect ratio.
    - Rotates images by a specified angle.
    - Aligns and crops images based on predefined parameters.
    - Reads annotations from JSON files and visualizes them by drawing bounding boxes on images.
    - Save the processed and annotated images to specified filenames.

    Key Features:
    - Resizes images to fit within a target size.
    - Rotates images to correct orientation.
    - Aligns and crops images based on central positioning.
    - Annotates images with bounding boxes from JSON files.
    - Blends and saves annotated images.

Usage:
    1. Update the `dataset` and `folder` variables with the appropriate dataset path and folder name.
    2. Run the script to process and align images, and annotate them with bounding boxes.

Requirements:
    - OpenCV
    - NumPy
    - JSON
    - OS
"""

import os
import cv2
import numpy as np
import json


# for f in sorted(os.listdir()):
#     if f.endswith('.json'):
#         with open(f, 'r') as f_:
#             dataset = json.load(f_)
#         filepath = dataset['imagePath']
#         filepath = filepath.split('_')
#         del (filepath[-1])
#         filepath.append('W.jpg')
#         filepath = "_".join(filepath)
#         dataset['imagePath'] = filepath
#         with open(f, 'w') as jsonfile:
#             json.dump(dataset, jsonfile)


def fit_image(image, target_width, target_height):
    height, width = image.shape[:2]

    # Compute scaling factors
    scale_x = target_width / width
    scale_y = target_height / height
    scale = min(scale_x, scale_y)

    # Resize the image
    new_width = int(width * scale)
    new_height = int(height * scale)
    scaling_matrix = np.float32([
        [scale, 0, 0],
        [0, scale, 0]
    ])

    scaled_image = cv2.warpAffine(image, scaling_matrix, (new_width, new_height))

    # Create a blank canvas
    result_image = np.zeros((target_height, target_width, 3), dtype=np.uint8)

    # Compute position to center the scaled image
    x_offset = (target_width - new_width) // 2
    y_offset = (target_height - new_height) // 2

    result_image[y_offset:y_offset + new_height, x_offset:x_offset + new_width] = scaled_image

    return result_image


def rotate_image(image, angle):
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result


def align_image():
    for img_W in sorted(os.listdir()):
        if img_W.endswith('W.jpg'):
            W = cv2.imread(img_W)
            ################
            img_T = img_W.replace('W.jpg', 'T.jpg')
            T = cv2.imread(img_T)
            ################
            W = cv2.resize(W, (1387, 780))
            # W = rotate_image(W, 0.38050638)
            height, width, depth = W.shape
            print(width, height)
            x_center = width // 2 - 17
            y_center = height // 2 - 4
            ####################################
            # angle = np.radians(-3)
            # perspective_matrix = np.float32([
            #     [1, np.sin(angle) * 0.5, 0],
            #     [0, 1, 0],
            #     [0, 0, 1]
            # ])
            ####################################
            x_1 = x_center - 640 // 2
            y_1 = y_center - 512 // 2
            x_2 = x_center + 640 // 2
            y_2 = y_center + 512 // 2

            print(x_1, y_1)
            print(x_2, y_2)

            cropped_img = W[y_1:y_2, x_1:x_2]
            cropped_img = cv2.resize(cropped_img, (640, 512))

            # cropped_img = cv2.warpPerspective(cropped_img, perspective_matrix, (640, 512))
            # cropped_img = fit_image(cropped_img,640, 512)
            # T = cv2.resize(T, (640, 512))
            # cv2.imshow(img_W, cropped_img)
            # cv2.imshow(img_T, T)
            # blended = cv2.addWeighted(T, 0.8, cropped_img, 0.8, 0)
            # cv2.imshow('blend', blended)
            # cv2.waitKey()
            # cv2.destroyAllWindows()
            cv2.imwrite(img_W, cropped_img)


def read_annotation(jsonfile):
    with open(jsonfile, 'r') as jsonF:
        data = json.load(jsonF)

    shapes = data['shapes']
    W = None
    img_file = data['imagePath']
    img_T = img_file
    T = cv2.imread(img_T)

    filepath = img_T.split('_')
    del (filepath[-1])
    filepath.append('W.JPG')
    img_W = "_".join(filepath)
    if os.path.isfile(img_W):
        print(img_W)
        W = cv2.imread(img_W)

    filepath = img_T.split('_')
    filepath[-1] = str('W.jpg')
    img_W = "_".join(filepath)
    print(img_W)
    W = cv2.imread(img_W)
    blended = cv2.addWeighted(W, 0.8, T, 0.8, 0)

    for b in shapes:
        box = b['points']
        xy1, xy2 = box
        x1, y1 = xy1
        x2, y2 = xy2
        cv2.rectangle(T, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.rectangle(W, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.rectangle(blended, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(blended, 'person',
                    (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0),
                    2)
    # cv2.imshow(img_T, T)
    # cv2.imshow(img_W, W)
    cv2.imshow(img_W, blended)
    cv2.imwrite('annotated_' + img_T, T)
    cv2.imwrite('annotated_' + img_W, W)
    cv2.imwrite('blended' + img_W, blended)
    # cv2.waitKey()
    # cv2.destroyAllWindows()


if __name__ == "__main__":
    dataset = '/home/abdalraheem/Documents/GitHub/TEMA_Visual_IR_DL_Detection/datasets/Smoke_dataset'
    folder = 'DJI_202408011319_031/valid_thermal_images/2'
    print(os.path.join(dataset, folder))
    os.chdir(os.path.join(dataset, folder))

    # for f in sorted(os.listdir()):
    #     if f.endswith('.json'):
    #         read_annotation(f)
    align_image()
