import tensorflow as tf 
from tensorflow import keras
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import numpy as np 

(X_train, y_train), (X_test, y_test) = mnist.load_data()
y_train = to_categorical(y_train)
X_train = X_train.reshape(-1,28,28,1)/255
X_test = X_test.reshape(-1,28,28,1)/255
# print(y_test)
# print(X_train.shape)

model= keras.models.load_model('saved_model/my_model')


extractor = keras.Model(inputs=model.inputs,
                        outputs=[model.layers[3].output])
features = extractor(X_test[0].reshape(-1,28,28,1))

w1= (np.asarray(model.layers[0].get_weights()))
w2= (np.asarray(model.layers[2].get_weights()))
w3= (np.asarray(model.layers[5].get_weights()))

# print(w1)
# print(w2)
# print(w3)

print(model.layers[2].output.shape)



X_test = X_test.reshape(-1,28,28,1)
outs= model.predict(X_test)
result= np.argmax(outs, axis=1)

# print(outs)
score=0
for i in range(len(outs)):
    if result[i] == y_test[i]:
        score +=1
    # else:
    #     print(outs[i])
    #     print(y_test[i])


print("accuracy= ",score/len(outs)*100)