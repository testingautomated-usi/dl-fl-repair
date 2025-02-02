text = f"""
import time

start = time.time()
import keras
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from keras.losses import sparse_categorical_crossentropy
from keras.optimizers import Adam
import matplotlib.pyplot as plt
import os
import numpy as np
from sklearn.model_selection import train_test_split


def main():
    # Model configuration
    batch_size = 64
    img_width, img_height, img_num_channels = 32, 32, 3
    loss_function = sparse_categorical_crossentropy
    no_classes = 10
    no_epochs = 50
    optimizer = Adam()
    validation_split = 0.2
    verbosity = 1
    # Load CIFAR-10 data
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    # Determine shape of the data
    input_shape = (img_width, img_height, img_num_channels)
    # Parse numbers as floats
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    # Normalize data
    x_train = x_train / 255
    x_test = x_test / 255

    x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=validation_split, random_state=0)
    x_train_orig = np.copy(x_train);
    y_train_orig = np.copy(y_train);

    # Create the model
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, kernel_size=(3, 3), activation='hard_sigmoid'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(no_classes, activation='softmax'))

    # Compile the model
    model.compile(loss=loss_function,
                  optimizer=optimizer,
                  metrics=['accuracy'])
    for i, x in enumerate(model.layers):
        print(i, x)
    # Fit data to model
    history = model.fit(x_train, y_train,
                batch_size=batch_size,
                epochs=no_epochs,
                verbose=verbosity,
                validation_data=(x_valid, y_valid))

    # Generate generalization metrics
    score = model.evaluate(x_test, y_test, verbose=0)

    print('test data shape', x_test.shape)
    return score


if __name__ == '__main__':
    main()


end = time.time()
print(end - start)

"""

task = "classification problem"
dataset = "CIFAR-10 dataset"
