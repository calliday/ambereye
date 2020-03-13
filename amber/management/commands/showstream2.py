from django.core.management.base import BaseCommand, CommandError

# import the necessary packages
# from picamera.array import PiRGBArray
# from picamera import PiCamera

import picamera
import picamera.array
import time
import cv2
import imutils
import numpy as np


class Command(BaseCommand):
    help = 'Start capturing images'
    
    def handle(self, **options):    
        # initialize the camera and grab a reference to the raw camera capture
        with picamera.PiCamera() as camera:
#             camera.resolution = (2592, 1944)
            res = (2592, 1936)
            camera.resolution = res
#             camera.resolution = (2000, 1500)
#             camera.resolution = (1280, 720)
#             camera.resolution = (720, 540)
            camera.start_preview()
            time.sleep(2)
            with picamera.array.PiRGBArray(camera, size=res) as stream:
                camera.capture(stream, format='bgr')
                
                # At this point the image is available as stream.array
                frame = stream.array
        
        frame = imutils.rotate(frame, angle=180)
        cv2.imshow("frame", frame)
        cv2.waitKey(0);
        
#         camera = PiCamera()
#         camera.resolution = (640, 480)
#         camera.framerate = 32
#         rawCapture = PiRGBArray(camera, size=(640, 480))
#          
#         # allow the camera to warmup
#         time.sleep(0.1)
#          
#         # capture frames from the camera
#         for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
#             # grab the raw NumPy array representing the image, then initialize the timestamp
#             # and occupied/unoccupied text
#             image = frame.array
#          
#             # show the frame
#             cv2.imshow("Frame", image)
#             key = cv2.waitKey(1) & 0xFF
#          
#             # clear the stream in preparation for the next frame
#             rawCapture.truncate(0)
#          
#             # if the `q` key was pressed, break from the loop
#             if key == ord("q"):
#                 break
