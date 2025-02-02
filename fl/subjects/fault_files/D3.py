text = f"""

import time

import sklearn.model_selection
from keras.models import Sequential
from keras.layers import Dense, Activation
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
import keras


def main():
    dataset = pd.read_csv("./dataset/data.csv")
    X = dataset.values[:, 0].astype(float)
    Y = dataset.values[:, 1]
    scaler = StandardScaler()
    X = scaler.fit_transform(X.reshape(-1, 1))
    encoder = LabelEncoder()
    Y_encoded = encoder.fit_transform(Y)
    Y_onehot = np_utils.to_categorical(Y_encoded)

    X_train, X_test, Y_train, Y_test = sklearn.model_selection.train_test_split(X, Y_onehot, test_size=0.2,
                                                                                random_state=42, shuffle=True)
    
    # X_test_used, X_test_holdout, y_test_used, y_test_holdout = sklearn.model_selection.train_test_split(X_test, Y_test, test_size=0.2, random_state=42)

    model = Sequential()
    model.add(Dense(1, input_dim=1))
    model.add(Activation('sigmoid'))
    model.add(Dense(2, kernel_initializer='normal', activation='softmax'))
    model.compile(loss='mean_absolute_error', optimizer="rmsprop", metrics=['accuracy'])
    model.fit(X_train, Y_train, epochs=10, batch_size=200, verbose=0)
    results = model.evaluate(X_test, Y_test)
    print(results)

if __name__ == "__main__":
    main()

"""

task = "classification problem",
dataset = "artificially generated dataset"
