from django.core.management.base import BaseCommand, CommandError

# import the necessary packages
from amber.models import Car, CarPlacement
# from amber.dcolor_detect import get_colors, white_balance
from amber.dyolo import run

# import the necessary packages
from imutils.video import VideoStream
from collections import deque
import numpy as np
import argparse
import imutils
import time
import cv2
import os


class Command(BaseCommand):
    help = 'Start capturing images'
    
    def handle(self, **options):
#         color = "White"
#         car, _cre = Car.objects.get_or_create(
#             color=color,
#             license_plate="1BAE363",
#             style="sed"
#         )
#         placements = CarPlacement.objects.filter(car=car)
#         print(placements)
        run()