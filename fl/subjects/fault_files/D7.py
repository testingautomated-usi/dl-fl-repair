text = f"""
 from keras.models import Sequential
from keras.layers import Dense
from numpy import linspace
from sklearn.model_selection import train_test_split


# Generate dummy data
data = data = linspace(1,2,100).reshape(-1,1)
y = data*5
X_train, X_test, y_train, y_test = train_test_split(data, y, test_size=0.2, random_state=42)

# Define the model
def baseline_model():
    model = Sequential()
    model.add(Dense(1, activation = 'linear', input_dim = 1))
    model.compile(optimizer = 'rmsprop', loss = 'mean_squared_error',metrics = ['mae'])
    return model


# Use the model
regr = baseline_model()
regr.fit(X_train,y_train,epochs =200,batch_size = 32)
results = regr.evaluate(X_test, y_test)

print(results)

"""

task = "regression problem"
dataset = "artificial dummy dataset"
