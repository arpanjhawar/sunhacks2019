import cv2
import os
import tensorflow as tf
from keras import Sequential


cat = ["NORMAL", "PNEUMONIA"]


def prepare(file):
    imsz = 200
    img_array = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
    new_array = cv2.resize(img_array, (imsz, imsz))
    return new_array.reshape(-1, imsz, imsz, 1)


model = tf.keras.models.load_model("CNN4.model")
data_path = "chest_xray/chest_xray/test"
for a in cat:
    print("Hello world")
    path = os.path.join(data_path, a)
    for img in os.listdir(path):
        iarr = os.path.join(path, img)
        image = prepare(iarr)
        prediction = model.predict([image])
        prediction = list(prediction[0])
        print(cat[prediction.index(max(prediction))])
