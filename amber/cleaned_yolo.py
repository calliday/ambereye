# USAGE
# python yolo_video.py --input videos/airport.mp4 --output output/airport_output.avi --yolo yolo-coco

# import the necessary packages
from amber.CustomThreads import ThreadedCamera
import numpy as np
import imutils
import cv2
import os
from tensorflow.keras.models import load_model

# custom functions
# from amber.color_detect import get_colors
from amber.models import Car, CarPlacement
from amber.lp.Main import main

COLOR_MODEL = load_model('model_keras54.h5')
COLORS = ['Black', 'Silver', 'Gray', 'White', 'Yellow', 'Blue',
                   'Red', 'Purple', 'Green', 'Brown', 'Tan', 'Orange']

# construct the argument parse and parse the arguments
def run():
    args = {"confidence": 0.5, "threshold": 0.3}

    # load the COCO class labels our YOLO model was trained on
    labelsPath = "amber/yolo/yolo-coco/coco.names"
    LABELS = open(labelsPath).read().strip().split("\n")

    # derive the paths to the YOLO weights and model configuration
    weightsPath = "amber/yolo/yolo-coco/yolov3.weights"
    configPath = "amber/yolo/yolo-coco/yolov3.cfg"

    # load our YOLO object detector trained on COCO dataset (80 classes)
    # and determine only the *output* layer names that we need from YOLO
    print("[INFO] loading YOLO from disk...")
    net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
    ln = net.getLayerNames()
    ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    (W, H) = (None, None)

    # Start camera I/O on another thread
    cam = ThreadedCamera().start()

    # loop over frames from the video file stream
    while True:
        print("[INFO] new frame...")
        # key listener for q to quit the program
        # if the 'q' key is pressed, stop the loop
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            cam.stop()
            break

        # read the next frame from the stream
        frame = cam.read()

        frame = imutils.rotate(frame, angle=180)

        # show the frame
#         cv2.imshow("frame", frame)
#         cv2.waitKey(0);

        # if the frame dimensions are empty, grab them
        if W is None or H is None:
            (H, W) = frame.shape[:2]

        # construct a blob from the input frame and then perform a forward
        # pass of the YOLO object detector, giving us our bounding boxes
        # and associated probabilities
        blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416),
            swapRB=True, crop=False)
        net.setInput(blob)
        layerOutputs = net.forward(ln)  # this is the bottleneck

        # initialize our lists of detected confidences,
        # and class IDs, respectively
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
                    confidences.append(float(confidence))
                    classIDs.append(classID)

                    # CUSTOM: crop each object TODO
                    cropped = frame[y:y + int(height), x:x + int(width)]
                    if cropped.shape[0] < 1 or cropped.shape[1] < 1:
                        continue

                    # try to find and read license plate
#                     lp = threading.Thread(target=main(cropped), args=(1,))
                    lp = main(cropped)
                    if lp is None:
                        lp = "000"

                    # get the color of the car
                    # color = get_colors(cropped, 3)
                    color = get_color(cropped)
                    print('color is:', color)

                    # log to the database
                    car, _created = Car.objects.get_or_create(
                        color=color,
                        license_plate=lp,
                        style="sed"
                    )
                    placement = CarPlacement.objects.create(
                        car=car,
                        latitude=0,
                        longitude=0
                    )
                    print("placement:", placement)

#                     cv2.imshow("Frame", cropped)
#                     cv2.waitKey(0);


def get_color(image):
    cars = np.array([cv2.resize(image, (150, 150), interpolation=cv2.INTER_CUBIC)])
    prediction = COLOR_MODEL.predict(cars)
    index = np.where(prediction[0] == 1)
    return COLORS[index[0][0]]
