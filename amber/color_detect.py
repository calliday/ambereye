import os
import cv2
from collections import Counter
from sklearn.cluster import KMeans
from amber.color_names import colors_ben

def get_colors(image, number_of_colors):
    modified_image = cv2.resize(image, (120, 90), interpolation = cv2.INTER_AREA)
    modified_image = modified_image.reshape(modified_image.shape[0]*modified_image.shape[1], 3)

    clf = KMeans(n_clusters = number_of_colors)
    labels = clf.fit_predict(modified_image)

    counts = Counter(labels)
    key = counts.most_common(1)[0][0]

    center_colors = clf.cluster_centers_

    car_color = center_colors[key]
    
    r = int(round(car_color[0] / 85))
    g = int(round(car_color[1] / 85))
    b = int(round(car_color[2] / 85))
        
    return colors_ben[r][g][b]
