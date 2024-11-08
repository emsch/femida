from tkinter import *
import PIL
from PIL import Image, ImageDraw
import numpy as np
import matplotlib.pyplot as plt
from keras.models import model_from_json
model = model_from_json(open("models/mnist_mega_model_4_sep_1_7.json").read())
model.load_weights('models/best_great_model_sep_1_7.hdf5')
current_recognized = None
import string
import random
import os

def randomString(stringLength=8):
    letters = string.ascii_lowercase
    return ''.join(random.choice(letters) for i in range(stringLength))


if not os.path.exists("paintings"):
    os.makedirs("paintings")

def save():
    global image_number,current_recognized
    filename = f'paintings/image_{image_number}_{randomString()}.png'
    image1.save(filename)
    np_image = np.array(image1.resize((28, 28), Image.ANTIALIAS))
    np_image = np.mean(np_image, axis=2)
    plt.imshow(np_image, "gray")
    filename = f'paintings/{current_recognized}_small_image_{image_number}_{randomString()}.png'
    plt.savefig(filename)
    image_number += 1

def clear():
    global cv, draw, image1
    cv.delete("all")
    image1 = PIL.Image.new('RGB', (500, 500), 'white')
    draw = ImageDraw.Draw(image1)

def activate_paint(e):
    global lastx, lasty
    cv.bind('<B1-Motion>', paint)
    lastx, lasty = e.x, e.y


def paint(e):
    global lastx, lasty,w, draw, model,current_recognized
    x, y = e.x, e.y
    cv.create_line((lastx, lasty, x, y), width=30)
    #  --- PIL
    draw.line((lastx, lasty, x, y), fill='black', width=30)
    lastx, lasty = x, y
    np_image = np.array(image1.resize((28, 28), Image.ANTIALIAS))
    np_image = np.mean(np_image, axis=2)
    prediction = model.predict(np_image.reshape(1, 28, 28, 1))
    predicted_digit = prediction.argmax()
    if predicted_digit == 10:
        predicted_digit = "Пустота"
    elif predicted_digit == 11:
        predicted_digit = "☑️"
    current_recognized = predicted_digit
    confidence = np.max(prediction)
    w["text"] = "Digit: {}, Conf:{}".format(predicted_digit, round(confidence, 2))



root = Tk()

lastx, lasty = None, None
image_number = 0

cv = Canvas(root, width=500, height=500, bg='white',bd=10, relief='ridge')
# --- PIL
image1 = PIL.Image.new('RGB', (500, 500), 'white')
draw = ImageDraw.Draw(image1)

cv.bind('<1>', activate_paint)
cv.pack(expand=YES, fill=BOTH)

btn_save = Button(text="save", command=save)
btn_save.pack()


btn_clear = Button(text="clear", command=clear)
btn_clear.pack()

w = Label(root, text="Здесь будет распознавание нарисованного числа")
w.pack()

root.mainloop()
