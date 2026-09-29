import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras import Sequential, layers

# 1) Import dataset
df = pd.read_csv('airline-passengers.csv')
data = df['Passengers'].values.reshape(-1, 1).astype('float32')
scaler = MinMaxScaler()
data = scaler.fit_transform(data)

# Make sequences: use the last 12 months to predict the next month
look_back = 12
X, y = [], []
for i in range(len(data) - look_back):
    X.append(data[i:i + look_back])
    y.append(data[i + look_back])
X, y = np.array(X), np.array(y)

# Split: first 80% train, last 20% test (no shuffling for time series)
split = int(len(X) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# 2) Train RNN, LSTM and GRU with identical settings
for name, Layer in [('RNN', layers.SimpleRNN), ('LSTM', layers.LSTM), ('GRU', layers.GRU)]:
    model = Sequential([
        layers.Input(shape=(look_back, 1)),
        Layer(50),
        layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    model.fit(X_train, y_train, epochs=100, batch_size=8, verbose=0)

    # 3) Predict and calculate MSE / RMSE (in real passenger numbers)
    pred = scaler.inverse_transform(model.predict(X_test))
    actual = scaler.inverse_transform(y_test)
    mse = mean_squared_error(actual, pred)
    print(name, "MSE:", mse, "RMSE:", np.sqrt(mse))
