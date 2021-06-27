import tkinter as tk
import tkcap
from PIL import Image
import tensorflow as tf
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.callbacks import Callback
from extra_keras_datasets.extra_keras_datasets import emnist
import numpy as np
from time import sleep


#NOTE: To draw a number on the canvas, you must draw it in one mouse stroke: left-click once to start drawing and once again to stop


class Callback(Callback): #callback class from predefined keras superclass: stops training once accuracy is 95%
    def on_epoch_end(self, epochs, logs={}):
        if(logs.get('accuracy') > 0.9):
            self.model.stop_training = True

CallbackTraining = Callback()

def turnVisible(event): #reads clicks of the mouse to enable users to draw a digit on the canvas (after first click) and "freeze" the canvas after the second screen
    global color, counter, root #also starts training of the NN after second click
    counter+=1
    if(color == "black"):
        color = "white"
    else:
        color = "black"
    if(counter >= 2):
        root.unbind('<Button-1>') #unbinds motion and left mouse clicking for drawing so that only the number can appear on the canvas
        root.unbind('<Motion>')
        cap = tkcap.CAP(root) #use of tkcap repo to screenshot the tkinter window
        cap.capture(r'C:\Users\Sahil\Data\Numbers\TkinterWindow.jpg', overwrite=True)
        root.destroy() #closes the drawing window, breaking root.mainloop() and starting the training of the NN

def myfunction(event): #function to track which is used in tandem with the tkinter motion listener to draw on the canvas
    x, y = event.x, event.y
    if canvas.old_coords:
        init_x, init_y = canvas.old_coords
        canvas.create_line((x, y, init_x, init_y), fill = color, width = 8)
    canvas.old_coords = x, y

def userSad(event): #falsifies while loop truth condition to end drawing session
    global userHappy
    userHappy = False

(td, tl), (testd, testl) = emnist.load_data(type = 'digits') #loads and reshapes the MNIST data set

td = td/255.0
testd = testd/255.0

td = td.reshape(240000, 28, 28, 1)
testd = testd.reshape(40000,28,28,1)

network = Sequential([ #creates a sequential CNN with various conv, pooling, dropout, and dense layers
    Conv2D(16, (3,3), activation = tf.nn.relu, input_shape= (28, 28, 1)),
    Dropout(0.3),
    Conv2D(32, (3,3), activation = tf.nn.relu),
    MaxPooling2D(2,2),
    Conv2D(32, (3,3,), activation = tf.nn.relu),
    Dropout(0.3),
    Conv2D(64,(3,3), activation = tf.nn.relu),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(100, activation = tf.nn.relu),
    Dense(10, activation = tf.nn.softmax) #softmax activation due to the desired action of multiclass classification
])

network.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy',metrics = ['accuracy']) #compiles and starts learning of the network
network.fit(td, tl, epochs = 20, callbacks = [CallbackTraining])
network.evaluate(testd, testl) #evaluates the performance of the network using the MNIST test data

userHappy = True

while(userHappy): #user can repeatedly draw digits until the program is stopped

    color = "black"
    counter = 0

    root = tk.Tk() #creates a tkinter object

    canvas = tk.Canvas(root, bg ='black', width=400, height=400) #creates tkinter canvas object of size 400x400 with black background
    canvas.pack()
    canvas.old_coords = None


    root.bind('<Motion>', myfunction) #binds mouse movement for drawing and mouse left-clicking for starting/stopping drawing
    root.bind('<Button-1>', turnVisible)
    root.bind('<Button-3>', userSad)
    root.mainloop() #allows binds to act on canvas until it is closed

    # Opens a image in RGB mode
    im = Image.open(r'C:\Users\Sahil\Data\Numbers\TkinterWindow.jpg')
    original_file_path = r'C:\Users\Sahil\Data\Numbers\TkinterWindow.jpg'

    width, height = im.size

    left = 5
    top = 50
    right = width
    bottom = height

    im1 = im.crop((left, top, right, bottom)) #crops screenshot of tkinter window to exclude the top 'label-bar'

    im1 = im1.resize((28,28)) #resizes the cropped screenshot to a format exceptable by the neural network and saves it locally
    im1.save('resizedTkinter.jpg')


    im = Image.open('resizedTkinter.jpg').convert('L') #opens the screenshot of the cropped canvas drawing as grayscale image
    im1 = im.resize((28, 28))#resizes/reshapes the image for compatibility with the NN
    im1 = np.reshape(im,[1, 28, 28, 1])

    array = max(network.predict(im1))

    print('The number is ', np.argmax(array)) #prints the classification of the number as a digit from 0-9
    sleep(1) #sets a delay so that users are not bombarded by tkinter popup window
