from sklearn.cluster import KMeans  # sklearn
import matplotlib.pyplot as plt     # matplotlib
import numpy as np
import cv2
from collections import Counter
from skimage.color import rgb2lab, deltaE_cie76  # scikit-image
import os

colors = {
	'red': (255,0,0),
	'dark red': (63,0,0),
	'gold': (212,175,55),
	'brown': (150,75,0),
	'green': (0,255,0),
	'dark green': (0,63,0),
	'blue': (0,0,255),
	'dark blue': (0,0,63),
	'yellow': (255,255,0),
	'orange': (255,127,0),
	'white': (255,255,255),
	'silver': (170,170,170),
	'black': (0,0,0),
	'gray': (85,85,85),
	'pink': (255,127,127),
	'purple': (127,0,255),
}

def distance(left, right):
	return sum((l-r)**2 for l, r in zip(left, right))**0.5

class NearestColorKey(object):
	def __init__(self, goal):
		self.goal = goal
	def __call__(self, item):
		return distance(self.goal, item[1])

def RGB2HEX(color):
	return "#{:02x}{:02x}{:02x}".format(int(color[0]), int(color[1]), int(color[2]))

def get_image(image_path):
	image = cv2.imread(image_path)
	image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	return image
	
def get_colors(image, number_of_colors, show_chart):
	modified_image = cv2.resize(image, (120, 90), interpolation = cv2.INTER_AREA)
	modified_image = modified_image.reshape(modified_image.shape[0]*modified_image.shape[1], 3)

	clf = KMeans(n_clusters = number_of_colors)
	labels = clf.fit_predict(modified_image)
	
	counts = Counter(labels)
	##print("counts:", counts)
	key = counts.most_common(1)
	key = key[0][0]
	##print("key:", key)

	center_colors = clf.cluster_centers_
	##print("center_colors:", center_colors)
	##print("center_colors[0]:", center_colors[key])
	# We get ordered colors by iterating through the keys
	ordered_colors = [center_colors[i] for i in counts.keys()]
	#print(ordered_colors)
	hex_colors = [RGB2HEX(ordered_colors[i]) for i in counts.keys()]
	rgb_colors = [ordered_colors[i] for i in counts.keys()]
	
	car_color = center_colors[key]
	##print(car_color)
	##print(hex_colors[0])
	r = car_color[0]
	g = car_color[1]
	b = car_color[2]
	
	print(min(colors.items(), key=NearestColorKey((r, g, b)))[0])
		
	if (show_chart):
		plt.figure(figsize = (8, 6))
		plt.pie(counts.values(), labels = hex_colors, colors = hex_colors)
		plt.show()
	return rgb_colors

#get_colors(get_image('car_cropped.jpg'), 3, True)
	
