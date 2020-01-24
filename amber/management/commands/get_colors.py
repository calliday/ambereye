from django.core.management.base import BaseCommand, CommandError

# import the necessary packages
from sklearn.cluster import KMeans  # sklearn
import matplotlib.pyplot as plt     # matplotlib
import numpy as np
import cv2
from collections import Counter
from skimage.color import rgb2lab, deltaE_cie76  # scikit-image
import os

from amber.models import Car


def RGB2HEX(color):
	return "#{:02x}{:02x}{:02x}".format(int(color[0]), int(color[1]), int(color[2]))

def get_image(image_path):
	image = cv2.imread(image_path)
	image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	return image
	
def get_colors(image, number_of_colors, show_chart):
	modified_image = cv2.resize(image, (60, 40), interpolation = cv2.INTER_AREA)
	modified_image = modified_image.reshape(modified_image.shape[0]*modified_image.shape[1], 3)

	clf = KMeans(n_clusters = number_of_colors)
	labels = clf.fit_predict(modified_image)
	
	counts = Counter(labels)

	center_colors = clf.cluster_centers_
	# We get ordered colors by iterating through the keys
	ordered_colors = [center_colors[i] for i in counts.keys()]
	hex_colors = [RGB2HEX(ordered_colors[i]) for i in counts.keys()]
	rgb_colors = [ordered_colors[i] for i in counts.keys()]

	if (show_chart):
		plt.figure(figsize = (8, 6))
		plt.pie(counts.values(), labels = hex_colors, colors = hex_colors)
		plt.show()
	return rgb_colors


class Command(BaseCommand):
	help = 'Get prevalent colors in image'

	def add_arguments(self, parser):
		#parser.add_argument('jpg_file', help="image of car")
		parser.add_argument('num_colors', type=int, help="number of colors to display")
	
	def handle(self, **options):	
		# get_colors(get_image('./car_cropped.jpg'), options['num_colors'], True)
		self.car, _cre = Car.objects.get_or_create(color="white", license_plate="000", style="suv")
		
