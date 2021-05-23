import numpy as np 
import tensorflow as tf 
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Dense, Conv2D, Flatten, ReLU, Softmax 
from tensorflow.keras.layers.experimental.preprocessing import Rescaling
from tensorflow.keras.models import Sequential
from tensorflow.keras import backend as K
from tensorflow.keras.constraints import NonNeg

class Convint(keras.constraints.Constraint):

  def __init__(self):
      pass

  def __call__(self,w):
    return tf.math.round(w)


(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train/100
X_test = X_test/100

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
X_train = X_train.reshape(-1,28,28,1)
X_test = X_test.reshape(-1,28,28,1)

print(X_train.shape)

model = Sequential()

model.add(Conv2D(3, kernel_size=(4,4), input_shape=(28,28,1),activation='relu'))
model.add(Rescaling(scale=0.1))
model.add(Conv2D(2, kernel_size=(3,3), activation='relu'))
model.add(Rescaling(scale=0.1))

model.add(Flatten())
model.add(Dense(10))
model.add(Rescaling(scale=0.1))

model.add(Softmax())

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train,y_train,validation_data=(X_test,y_test),epochs=1)

w1= (np.asarray(model.layers[0].get_weights())*10)
# print(w1)
w1[0]=np.around(w1[0])
w1[1]=np.around(w1[1])

w2= (np.asarray(model.layers[2].get_weights())*10)
w2[0]=np.around(w2[0])
w2[1]=np.around(w2[1])
print(w2[1].shape)
 
w3= (np.asarray(model.layers[5].get_weights())*10) 
# print(w3)
w3[0]=np.around(w3[0])
w3[1]=np.around(w3[1])


model.layers[5].set_weights(w3)
model.layers[0].set_weights(w1)
model.layers[2].set_weights(w2)


model.save('saved_model/my_model')


