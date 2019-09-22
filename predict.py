from tkinter import *
from tkinter import ttk
from tkinter import filedialog
import cv2
from PIL import ImageTk, Image
import os
import tensorflow as tf
from keras import Sequential


def chooseip():
    root = Tk()
    root.withdraw()
    global filename
    filename = filedialog.askopenfilename(
        parent=root,
        filetypes=[("Image Files", "*.jpeg")],
        title='Choose an Input Image'
    )
    if filename != None:
        global mainFile
        mainFile = filename


def predict():
    cat = ["NORMAL", "PNEUMONIA"]

    def prepare(file):
        imsz = 200
        img_array = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
        new_array = cv2.resize(img_array, (imsz, imsz))
        return new_array.reshape(-1, imsz, imsz, 1)

    model = tf.keras.models.load_model("CNN3.model")
    data_path = filename
    image = prepare(data_path)
    prediction = model.predict([image])
    prediction = list(prediction[0])
    print(cat[prediction.index(max(prediction))])
    Result.set(cat[prediction.index(max(prediction))])


root = Tk()
root.title("Image Classificaiton")

mainframe = ttk.Frame(root, padding="3 3 12 12")
mainframe.grid(column=0, row=0, sticky=(N, W, E, S))
root.columnconfigure(0, weight=1)
root.rowconfigure(0, weight=1)

Result = StringVar()

ttk.Button(mainframe, text="Choose Input Image File", command=chooseip).grid(column=1, row=1, sticky=W)

ttk.Label(mainframe, textvariable=Result).grid(column=3, row=3, sticky=(W, E))

ttk.Button(mainframe, text="Classify", command=predict).grid(column=3, row=1, sticky=W)

ttk.Label(mainframe, text="Result:").grid(column=1, row=3, sticky=W)

for child in mainframe.winfo_children(): child.grid_configure(padx=5, pady=5)

root.bind('<Return>', predict)

root.mainloop()
