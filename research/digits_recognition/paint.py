from tkinter import *
import PIL
from PIL import Image, ImageDraw
import numpy as np
import matplotlib.pyplot as plt
from keras.models import model_from_json
model = model_from_json(open("mnist_mega_model_2.json").read())
model.load_weights('mnist_mega_model_2')

def save():
    global image_number
    filename = f'middle_paintings/image_{image_number}.png'   # image_number increments by 1 at every save
    image1.save(filename)
    np_image = np.array(image1.resize((28, 28), Image.ANTIALIAS))
    np_image = np.mean(np_image, axis=2)
    plt.imshow(np_image, "gray")
    filename = f'middle_paintings/small_image_{image_number}.png'
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
    global lastx, lasty,w, draw, model
    x, y = e.x, e.y
    cv.create_line((lastx, lasty, x, y), width=30)
    #  --- PIL
    draw.line((lastx, lasty, x, y), fill='black', width=30)
    lastx, lasty = x, y
    np_image = np.array(image1.resize((28, 28), Image.ANTIALIAS))
    np_image = np.mean(np_image, axis=2)
    prediction = model.predict(np_image.reshape(1, 28, 28, 1))
    predicted_digit = prediction.argmax()
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
