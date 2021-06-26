import tkinter as tk
import tkcap
from PIL import Image
import tensorflow as tf
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.datasets import mnist
import numpy as np


class Callback(Callback):
    def on_epoch_end(self, epochs, logs={}):
        if(logs.get('accuracy') > 0.95):
            self.model.stop_training = True

CallbackTraining = Callback()

def turnVisible(event):
    global color, counter, root
    counter+=1
    if(color == "black"):
        color = "white"
    else:
        color = "black"
    if(counter >= 2):
        root.unbind('<Button-1>')
        root.unbind('<Motion>')
        cap = tkcap.CAP(root)
        cap.capture(r'C:\Users\Sahil\Data\Numbers\TkinterWindow.jpg', overwrite=True)
        root.destroy()

def myfunction(event):
    x, y = event.x, event.y
    if canvas.old_coords:
        init_x, init_y = canvas.old_coords
        canvas.create_line((x, y, init_x, init_y), fill = color, width = 8)
    canvas.old_coords = x, y


color = "black"
counter = 0

root = tk.Tk()

canvas = tk.Canvas(root, bg ='black', width=400, height=400)
canvas.pack()
canvas.old_coords = None


root.bind('<Motion>', myfunction)
root.bind('<Button-1>', turnVisible)

root.mainloop()

print('moving on')

# Opens a image in RGB mode
im = Image.open(r'C:\Users\Sahil\Data\Numbers\TkinterWindow.jpg')
original_file_path = r'C:\Users\Sahil\Data\Numbers\TkinterWindow.jpg'

width, height = im.size

left = 5
top = 50
right = width
bottom = height

im1 = im.crop((left, top, right, bottom))
#im1.show()

im1 = im1.resize((28,28))
im1.save('resizedTkinter.jpg')

(td, tl), (testd, testl) = mnist.load_data()

td = td/255.0
testd = testd/255.0

td = td.reshape(60000, 28, 28, 1)
testd = testd.reshape(10000,28,28,1)

network = Sequential([
    Conv2D(16, (3,3), activation = tf.nn.relu, input_shape= (28, 28, 1)),
    MaxPooling2D(2,2),
    Conv2D(32, (3,3), activation = tf.nn.relu),
    MaxPooling2D(2,2),
    Conv2D(64,(3,3), activation = tf.nn.relu),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(100, activation = tf.nn.relu),
    Dense(10, activation = tf.nn.softmax)
])

network.summary()

network.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy',metrics = ['accuracy'])
network.fit(td, tl, epochs = 20, callbacks = [CallbackTraining])
network.evaluate(testd, testl)

im = Image.open('resizedTkinter.jpg').convert('L')
im1 = im.resize((28, 28))
im1 = np.reshape(im,[1, 28, 28, 1])

array = max(network.predict(im1))
print('The number is ', np.argmax(array))
