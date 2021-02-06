from tensorflow.keras.datasets import boston_housing
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense
from sklearn import preprocessing
# load the data
(x_train, y_train), (x_test, y_test) = boston_housing.load_data()

# preprocess the data
x_train_scaled = preprocessing.scale(x_train)
scaler = preprocessing.StandardScaler().fit(x_train)
# x_test_scaled = preprocessing.StandardScaler.transform(x_test)

model = Sequential()
model.add(Dense(64, activation='relu', kernel_initializer='normal', input_shape=(13, )))
model.add(Dense(64, activation='relu'))
model.add(Dense(1))

model.compile(optimizer='rmsprop', loss='mse', metrics=['mean_absolute_error'])

model.fit(x_train_scaled, y_train, epochs=200, batch_size=128, callbacks=[EarlyStopping(monitor='loss', patience=20)])
