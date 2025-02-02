text = f"""
import pickle
# from Utils.utils import save_pkl, pack_train_config
from sklearn.model_selection import train_test_split
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
import keras

RADIX = 7

np.random.seed(42)


def _get_number(vector):
    return sum(x * 2 ** i for i, x in enumerate(vector))


def _get_mod_result(vector):
    return _get_number(vector) % RADIX


def _number_to_vector(number):
    binary_string = bin(number)[2:]
    if len(binary_string) > 20:
        raise NotImplementedError
    bits = (((0,) * (20 - len(binary_string))) +
            tuple(map(int, binary_string)))[::-1]
    assert len(bits) == 20
    return np.c_[bits]


def get_mod_result_vector(vector):
    return _number_to_vector(_get_mod_result(vector))


X = np.random.randint(2, size=(10000, 20))
Y = np.vstack(map(get_mod_result_vector, X))

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
# X_test_used, X_test_holdout, y_test_used, y_test_holdout = train_test_split(
#         x_test, y_test, test_size=0.2, random_state=42)

model = Sequential()
model.add(Dense(units=20, activation='relu', input_dim=20))
model.add(Dense(units=20, activation='relu'))
model.add(Dense(units=20, activation='softmax'))

model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(x_train, y_train, epochs=10, batch_size=50, verbose=0)

results = model.evaluate(x_test, y_test)

print(results)
"""

task = "classification problem"
dataset = "randomly generated dataset"
