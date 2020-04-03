import csv
import cv2
import numpy as np

from tensorflow.keras.models import load_model, model_from_json
# from tensorflow.keras.datasets import mnist
# from tensorflow.keras.utils import to_categorical

possible_colors = ['Black', 'Silver', 'Gray', 'White', 'Yellow', 'Blue',
                   'Red', 'Purple', 'Green', 'Brown', 'Tan', 'Orange']

print("load csv")
boxes = {}
with open('labeler/boxes.csv') as handle:
    reader = csv.DictReader(handle)
    for row in reader:
        boxes['./labeler/' + row['img'].strip()] = (row['x1'], row['y1'], row['x2'], row['y2'])

color_model = load_model('model_keras54.h5')

print("load images")
images = ['./labeler/car_ims/{:06d}.jpg'.format(x) for x in range(3001, 3101)]

output = []
for image in images:
    x1, y1, x2, y2 = boxes[image]
    frame = cv2.imread(image, cv2.IMREAD_COLOR)
    cropped = frame[int(y1):int(y2), int(x1):int(x2)]
    cars = np.array([cv2.resize(cropped, (150, 150), interpolation=cv2.INTER_CUBIC)])
    prediction = color_model.predict(cars)
    index = np.where(prediction[0] == 1)
    output.append(possible_colors[index[0][0]])
# print(index[0][0])
print(output)
# (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
# test_images = test_images.reshape((10000, 28, 28, 1))
# test_images = test_images.astype('float32') / 255
# test_labels = to_categorical(test_labels)
#
# # load json and create model
# json_file = open('model.json', 'r')
# loaded_model_json = json_file.read()
# json_file.close()
# loaded_model = model_from_json(loaded_model_json)
# # load weights into new model
# loaded_model.load_weights("model.h5")
# print("Loaded model from disk")
#
# # evaluate loaded model on test data
# loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
# score = loaded_model.evaluate(test_images, test_labels, verbose=1)
# print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))
