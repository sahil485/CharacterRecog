from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.callbacks import Callback
import tensorflow as tf

class Callback(Callback): #callback class from predefined keras superclass: stops training once accuracy is 95%
    def on_epoch_end(self, epochs, logs={}):
        if(logs.get('accuracy') > 0.9):
            self.model.stop_training = True

CallbackTraining = Callback()

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
    Dense(27, activation = tf.nn.softmax) #softmax activation due to the desired action of multiclass classification
])

network.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy',metrics = ['accuracy']) #compiles and starts learning of the network
network.fit(td, tl, epochs = 20, callbacks = [CallbackTraining])
network.evaluate(testd, testl) #evaluates the performance of the network using the MNIST test data

#td, tl, testd, testl are training data, training labels, test data, and test labels, respectively