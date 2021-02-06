import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import SimpleRNN, Activation, Dropout, Dense, Reshape
from tensorflow.keras.models import Sequential

data = pd.read_csv('/home/tim/Datasets/Tesla/TSLA.csv')
# data.plot('Date', 'Adj Close')
# plt.show()
ts_data = data['Adj Close'].values.reshape(-1, 1)
print(ts_data.shape)
# get size of the train set needed

train_recs = int(len(ts_data) * 0.75)

# split data, 75% - 25%

train_data = ts_data[:train_recs]
test_data = ts_data[train_recs:]
# print(len(train_data), len(test_data))

# scale data btw 0-1
scaler = MinMaxScaler()
train_data = scaler.fit_transform(train_data)
test_data = scaler.fit_transform(test_data)


# format the data to get 'features' for each instance
# we need to define a lookback period --> i.e the number of days from the history that we want to use to predict the next value
# following function will return the target value of y i.e (stock price for a day) and x (values for each day in the lookback period)...


def get_lookback(inp, lookback):
    y = pd.DataFrame(inp)
    dataX = [y.shift(i) for i in range(1, lookback + 1)]
    dataX = pd.concat(dataX, axis=1)
    dataX.fillna(0, inplace=True)
    return dataX.values, y.values


look_back = 10
trainX, trainY = get_lookback(train_data, look_back)
testX, testY = get_lookback(test_data, look_back)
# print(trainX.shape) # (1812, 10)

# Build RNN
model = Sequential()
model.add(Reshape(target_shape=(look_back, 1), input_shape=(look_back, )))
model.add(SimpleRNN(32, input_shape=(look_back, 1)))
model.add(Dense(1))
model.add(Activation('linear'))
model.compile(loss='mse', optimizer='adam')

# model.fit(trainX, trainY, epochs=3, batch_size=1, validation_split=0.1)
# model.save('/home/tim/trained/tesla.h5')