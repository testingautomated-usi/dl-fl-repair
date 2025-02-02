text = f"""
from keras.layers import Dense, merge, Activation, Dropout
from keras.models import Model
from keras.models import Sequential
from keras.optimizers import SGD
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


def main():
    df = pd.read_csv("./training.csv")
    y = np.array(df['Label'].apply(lambda x: 0 if x=='s' else 1))
    X = np.array(df.drop(["EventId","Label"], axis=1))

    X_train, X_test, y_train, y_test = train_test_split(X[:10000], y[:10000], test_size=0.3, random_state=42)

    model = Sequential()

    model.add(Dense(600, input_shape=(31,),activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(400, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(100, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer='rmsprop',loss='binary_crossentropy',metrics=['accuracy'])
    model.fit(X_train,y_train,epochs=1,batch_size=1)

    results = model.evaluate(X_test, y_test)
    print(results)

if __name__ == '__main__':
    main()

"""

task = "classification problem"
dataset = "Higgs Boson challenge dataset"