import csv
import os
import cv2

ims = []

with open('labeler/targets.csv') as handle:
    reader = csv.DictReader(handle)

    for row in reader:
        ims.append('labler/car_ims/{}'.format(row['img'].strip()))

print('labeler/car_ims/000001.jpg')
print(ims[0])
print(cv2.imread(ims[0], cv2.IMREAD_COLOR).shape)
