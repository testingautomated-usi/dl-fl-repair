text = f"""
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
from keras.initializers import RandomNormal
from keras.utils import to_categorical
import sys
from sklearn.model_selection import train_test_split

(x_train, y_train), (x_test, y_test) = mnist.load_data()

img_rows, img_cols = 28, 28

x_train = x_train.reshape(x_train.shape[0], img_rows * img_cols)
x_test = x_test.reshape(x_test.shape[0], img_rows * img_cols)
input_shape = (img_rows * img_cols,)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

num_classes = 10
y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)

model = Sequential()
layer1 = Dense(30,
               input_shape=input_shape,
               kernel_initializer=RandomNormal(stddev=1),
               bias_initializer=RandomNormal(stddev=1))
model.add(layer1)
layer2 = Dense(10,
               kernel_initializer=RandomNormal(stddev=1),
               bias_initializer=RandomNormal(stddev=1))
model.add(layer2)
model.summary()
model.compile(optimizer=SGD(lr=3.0),
              loss='mean_squared_error',
              metrics=['accuracy'])

# Train
model.fit(x_train,
          y_train,
          batch_size=10,
          epochs=30,
          verbose=2)

results = model.evaluate(x_test, y_test)

print(results)
"""

task = "classification problem",

dataset = "MNIST dataset"
