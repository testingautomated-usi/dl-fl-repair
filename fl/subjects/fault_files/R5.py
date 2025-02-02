text = f"""
import time

start = time.time()
import keras
import os
from keras.datasets import reuters
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from sklearn.model_selection import train_test_split
import numpy as np

# "remove_validation_set",
# # "disable_batching",
# # "add_noise",
# # "delete_training_data",
# # "change_label",
# # "unbalance_train_data",
# # "make_output_classes_overlap",
# # "change_epochs"


def main():
    (x_train_init, y_train_init), (x_test, y_test) = reuters.load_data(num_words=None, test_split=0.2, seed=42)
    word_index = reuters.get_word_index(path="reuters_word_index.json")

    num_classes = max(y_train_init) + 1

    index_to_word = {{}}
    for key, value in word_index.items():
        index_to_word[value] = key

    max_words = 10000

    tokenizer = Tokenizer(num_words=max_words)
    x_train_init = tokenizer.sequences_to_matrix(x_train_init, mode='binary')
    x_test = tokenizer.sequences_to_matrix(x_test, mode='binary')

    y_train_init = keras.utils.to_categorical(y_train_init, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    x_train, x_val, y_train, y_val = train_test_split(x_train_init, y_train_init, test_size=0.2, random_state=42)

    x_train_orig = np.copy(x_train)
    y_train_orig = np.copy(y_train)

    model = Sequential()
    model.add(Dense(512, input_shape=(max_words,), activation ='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation ='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.SGD(), metrics=['accuracy'])
    print(model.metrics_names)
    batch_size = 32
    epochs = 3

    history = model.fit(x_train, y_train, validation_data=(x_val, y_val), batch_size=batch_size, epochs=epochs, verbose=1)

    score = model.evaluate(x_test, y_test, batch_size=batch_size, verbose=1)

    return score

if __name__ == '__main__':
    score = main()
    print(score)


end = time.time()
print(end - start)

"""

task = "classification problem"
dataset = "Reuters dataset"
