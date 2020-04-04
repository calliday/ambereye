# USAGE
# cd ~/ambereye
# python3 manage.py movidius

# import the necessary packages
from imutils.video import VideoStream
from collections import deque
import numpy as np
import argparse
import imutils
import time
import cv2
import csv
import os
import time
import picamera
import picamera.array
from tensorflow.keras.models import load_model

# custom functions
from amber.color_detect import get_colors
from amber.labeler.car_types_classes import possible_types
from amber.models import Car, CarPlacement
from amber.lp.Main import main
from amber.CustomThreads import ThreadedCamera

COLOR_MODEL = load_model('amber/model_keras61.h5')
TYPE_MODEL = load_model('amber/model_keras_types51.h5')
COLORS = ['Black', 'Silver', 'Gray', 'White', 'Yellow', 'Blue',
                   'Red', 'Purple', 'Green', 'Brown', 'Tan', 'Orange']
CAR_TYPES = {
    'SUV': 'suv',
    'Sedan': 'sed',
    'Coupe': 'coup',
    'Convertible': 'conv',
    'Semi': 'semi',
    'Truck': 'truck',
    'Van': 'van',
    'Motorcycle': 'moto',
    'Bus': 'bus'
}

# construct the argument parse and parse the arguments
def run():
    args = {"confidence": 0.5, "threshold": 0.3}

    # load the COCO class labels our YOLO model was trained on
    labelsPath = "amber/yolo/yolo-coco/coco.names"
    LABELS = open(labelsPath).read().strip().split("\n")

    # initialize a list of colors to represent each possible class label
    np.random.seed(42)
    COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),
        dtype="uint8")

    # derive the paths to the YOLO weights and model configuration
    weightsPath = "amber/yolo/yolo-coco/yolov3.weights"
    configPath = "amber/yolo/yolo-coco/yolov3.cfg"

    # load our YOLO object detector trained on COCO dataset (80 classes)
    # and determine only the *output* layer names that we need from YOLO
    print("[INFO] loading YOLO from disk...")
    net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

    # Custom
    # Is this really all we have to do to set the target CPU to Movidius?
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_MYRIAD)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_INFERENCE_ENGINE)

    ln = net.getLayerNames()
    ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    # allow camera to warm up
    time.sleep(2.0)

    writer = None
    (W, H) = (None, None)

    # Start camera I/O on another thread
    cam = ThreadedCamera().start()

    # loop over frames from the video file stream
    while True:
        print("new frame")
        # key listener for q to quit the program
        # if the 'q' key is pressed, stop the loop
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            cam.stop()
            break

        # read the next frame from the stream
        frame = cam.read()

        # if the frame dimensions are empty, grab them
        if W is None or H is None:
            (H, W) = frame.shape[:2]

        # construct a blob from the input frame and then perform a forward
        # pass of the YOLO object detector, giving us our bounding boxes
        # and associated probabilities
        blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416),
            swapRB=True, crop=False)
        net.setInput(blob)
        start = time.time()
        layerOutputs = net.forward(ln)
        end = time.time()

        # initialize our lists of detected bounding boxes, confidences,
        # and class IDs, respectively
        boxes = []
        confidences = []
        classIDs = []

        # loop over each of the layer outputs
        for output in layerOutputs:
            # loop over each of the detections
            for detection in output:
                # extract the class ID and confidence (i.e., probability)
                # of the current object detection
                scores = detection[5:]
                classID = np.argmax(scores)
                confidence = scores[classID]

                # CUSTOM: prevent other objects besides cars from being
                # processed
                if LABELS[classID] != 'car':
                    continue

                # filter out weak predictions by ensuring the detected
                # probability is greater than the minimum probability
                if confidence > args["confidence"]:
                    # scale the bounding box coordinates back relative to
                    # the size of the image, keeping in mind that YOLO
                    # actually returns the center (x, y)-coordinates of
                    # the bounding box followed by the boxes' width and
                    # height
                    box = detection[0:4] * np.array([W, H, W, H])
                    (centerX, centerY, width, height) = box.astype("int")

                    # use the center (x, y)-coordinates to derive the top
                    # and and left corner of the bounding box
                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))

                    # update our list of bounding box coordinates,
                    # confidences, and class IDs
                    boxes.append([x, y, int(width), int(height)])
                    confidences.append(float(confidence))
                    classIDs.append(classID)

                    # CUSTOM: crop each object TODO
                    cropped = frame[y:y + int(height), x:x + int(width)]
                    if cropped.shape[0] < 1 or cropped.shape[1] < 1:
                        continue


                    lp, found = main(cropped)
                    if not lp:
                        lp = "000"

                    # get the color of the car
                    # color = get_colors(cropped, 3)
                    color = get_color(cropped)
                    car_type = get_car_type(cropped)

                    car, _created = Car.objects.get_or_create(
                        color=color,
                        license_plate=lp,
                        style=CAR_TYPES[car_type]
                    )
                    placement = CarPlacement.objects.create(
                        car=car,
                        latitude=0,
                        longitude=0
                    )
                    print("placement:", placement)

                    cv2.imwrite("amber/static/img.jpg", cropped)

                    with open("amber/static/details.txt", 'w+') as details:
                        writer = csv.writer(details)
                        lines = [[]]
                        lines[0].append(color)
                        lines[0].append(found)
                        lines[0].append(lp)
                        writer.writerows(lines)

#                     cv2.imshow("Frame", cropped)
#                     cv2.waitKey(0)

#         # apply non-maxima suppression to suppress weak, overlapping
#         # bounding boxes
#         idxs = cv2.dnn.NMSBoxes(boxes, confidences, args["confidence"],
#             args["threshold"])
#
#         # ensure at least one detection exists
#         if len(idxs) > 0:
#             # loop over the indexes we are keeping
#             for i in idxs.flatten():
#                 # extract the bounding box coordinates
#                 (x, y) = (boxes[i][0], boxes[i][1])
#                 (w, h) = (boxes[i][2], boxes[i][3])
#
#                 # draw a bounding box rectangle and label on the frame
#                 color = [int(c) for c in COLORS[classIDs[i]]]
#                 cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
#                 text = "{}: {:.4f}".format(LABELS[classIDs[i]],
#                     confidences[i])
#                 cv2.putText(frame, text, (x, y - 5),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # release the file pointers
    print("[INFO] cleaning up...")
#     vs.stop()
    # vs.release()


def get_color(image):
    cars = np.array([cv2.resize(image, (150, 150), interpolation=cv2.INTER_CUBIC)])
    prediction = COLOR_MODEL.predict(cars)
    index = np.where(prediction[0] == 1)
    try:
        return COLORS[index[0][0]]
    except:
        return "Color Unknown"

def get_car_type(image):
    cars = np.array([cv2.resize(image, (150, 150), interpolation=cv2.INTER_CUBIC)])
    prediction = TYPE_MODEL.predict(cars)
    index = np.where(prediction[0] == 1)
    try:
        return possible_types[index[0][0]]
    except:
        return "Type Unknown"
