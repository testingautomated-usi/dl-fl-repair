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


def main(model_dir, model_name):
    #model_location = os.path.join('trained_models', model_name)
    model_location = os.path.join(model_dir, model_name)
    (x_train_init, y_train_init), (x_test, y_test) = reuters.load_data(num_words=None, test_split=0.2, seed=42)
    word_index = reuters.get_word_index(path="reuters_word_index.json")

    print('# of Training Samples: {}'.format(len(x_train_init)))
    print('# of Test Samples: {}'.format(len(x_test)))

    num_classes = max(y_train_init) + 1
    print('# of Classes: {}'.format(num_classes))# of Training Samples: 8982
    # of Test Samples: 2246
    # of Classes: 46
    index_to_word = {}
    for key, value in word_index.items():
        index_to_word[value] = key

    print(' '.join([index_to_word[x] for x in x_train_init[0]]))
    print(y_train_init[0])

    max_words = 10000

    tokenizer = Tokenizer(num_words=max_words)
    x_train_init = tokenizer.sequences_to_matrix(x_train_init, mode='binary')
    x_test = tokenizer.sequences_to_matrix(x_test, mode='binary')

    y_train_init = keras.utils.to_categorical(y_train_init, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)
    print(x_train_init[0])
    print(len(x_train_init[0]))

    print(y_train_init[0])
    print(len(y_train_init[0]))

    x_train, x_val, y_train, y_val = train_test_split(x_train_init, y_train_init, test_size=0.2, random_state=42)
    print(len(x_train))
    X_test_used, X_test_holdout, y_test_used, y_test_holdout = train_test_split(x_test, y_test, test_size=0.2, random_state=42)
    x_train_orig = np.copy(x_train)
    y_train_orig = np.copy(y_train)
    print("mdl", model_location)

    if not os.path.exists(model_location):
        model = Sequential()
        model.add(Dense(512, input_shape=(max_words,), activation ='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(num_classes, activation ='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(learning_rate=1.0), metrics=['accuracy'])
        print(model.metrics_names)
        batch_size = 32
        epochs = 3

        history = model.fit(x_train, y_train, validation_data=(x_val, y_val), batch_size=batch_size, epochs=epochs, verbose=1)
        #model.save(model_location)

        score = model.evaluate(X_test_holdout, y_test_holdout)
        with open("/home/hpo/data/DC_orig_holdout_acc.txt", "a") as f:
            f.write(f"{os.path.basename(__file__)}|{score[1]:.2f}\n")
        print(score)
        score = model.evaluate(x_test, y_test, batch_size=batch_size, verbose=1)
    else:
        model = keras.models.load_model(model_location)
        score = model.evaluate(x_test, y_test, verbose=0)

    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    return score

if __name__ == '__main__':
    score = main('', 'reuters_change_learning_rate_mutated0_MP_False_1.0.h5')
