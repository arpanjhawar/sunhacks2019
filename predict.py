import numpy as np
import os
from matplotlib import pyplot as plt
import cv2
import random
import pickle

file_list = []
class_list = []
data_path = "chest_xray/chest_xray/train"
# All the categories you want your neural network to detect
cat = ["NORMAL", "PNEUMONIA"]
imgsize = 200
for a in cat:
    path = os.path.join(data_path, a)
    for img in os.listdir(path):
        iarr = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
training_data = []
print("Hello World 2")


def create_training_data():
    for category in cat:
        pa = os.path.join(data_path, category)
        class_num = cat.index(category)
        for imag in os.listdir(pa):
            img_array = cv2.imread(os.path.join(pa, imag), cv2.IMREAD_GRAYSCALE)
            new_array = cv2.resize(img_array, (imgsize, imgsize))
            training_data.append([new_array, class_num])


print("Hello World 3")
create_training_data()
print("Hello World")
random.shuffle(training_data)
x = []
y = []
for f, l in training_data:
    x.append(f)
    y.append(l)
x = np.array(x).reshape(-1, imgsize, imgsize, 1)
pickle_out = open("X.pickle", "wb")
pickle.dump(x, pickle_out)
pickle_out.close()
print("Hello World 4")
pickle_out = open("y.pickle", "wb")
pickle.dump(y, pickle_out)
pickle_out.close()
print("Hello World 5")
pickle_in = open("X.pickle", "rb")
x = pickle.load(pickle_in)
print("Hello World 6")
