import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

data = pd.read_csv('/home/tim/Datasets/GOOG.csv', date_parser=True)

train_data = data[data['Date'] < '2020-01-01'].copy()
test_data = data[data['Date'] >= '2020-01-01'].copy()

train_ds = train_data.drop(['Date', 'Adj Close'], axis=1)
test_ds = test_data.drop(['Date', 'Adj Close'], axis=1)

scaler = MinMaxScaler()
train_ds = scaler.fit_transform(train_ds)

X_train = []
y_train = []
for i in range(60, train_ds.shape[0]):
    X_train.append(train_ds[i-60:i])
    y_train.append(train_ds[i, 0])

X_train, y_train = np.array(X_train), np.array(y_train)
print(X_train.shape) # 60

regressor = Sequential()
regressor.add(LSTM(60, activation='relu', return_sequences=True, input_shape=(X_train.shape[1], 5)))
regressor.add(Dropout(0.2))

regressor.add(LSTM(60, activation='relu', return_sequences=True))
regressor.add(Dropout(0.2))

regressor.add(LSTM(80, activation='relu', return_sequences=True))
regressor.add(Dropout(0.2))

regressor.add(LSTM(120, activation='relu'))
regressor.add(Dropout(0.2))

regressor.add(Dense(1, activation='linear'))

regressor.compile(optimizer='adam', loss='mean_squared_error')

regressor.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.1)