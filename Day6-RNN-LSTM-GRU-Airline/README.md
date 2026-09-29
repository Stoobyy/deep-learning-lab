# Day 6: RNN, LSTM and GRU for Time-Series Prediction (Airline Passengers)

## Question

Design and implement RNN, LSTM, and GRU models for time-series prediction using the Airline Passenger dataset. Compare the performance of the models using RMSE and prediction plots.

**Dataset:** Airline Passenger dataset
**Input:** Monthly airline passenger counts
**Output:** Predicted future passenger count

**Models**
- Simple RNN
- LSTM
- GRU

**Tasks**
- Train Simple RNN, LSTM and GRU models
- Use identical: sequence length (look-back window), epochs, batch size
- Predict passenger counts on test data
- Plot: Actual vs Predicted values (RNN, LSTM, GRU)
- Calculate: RMSE, MSE
- Compare: prediction accuracy, RMSE, training time
- Analyze: ability to capture temporal patterns, long-term dependency learning, computational complexity
- Identify the best-performing model for time-series forecasting

> **Note:** This is the minimal exam version. It trains all three models with identical settings, predicts on test data, and prints MSE and RMSE for each. The plots, training-time comparison and written analysis are left out on purpose.
>
> **Why no confusion matrix?** This is a **regression** task: the model predicts a number (passenger count), not a class. A confusion matrix only works for classification, so MSE and RMSE are used instead.

## Files

- `rnn_lstm_gru_airline.py`: the code
- `airline-passengers.csv`: the dataset (144 monthly totals, Jan 1949 to Dec 1960, in thousands of passengers)

## How to run

```
pip install tensorflow scikit-learn pandas
python rnn_lstm_gru_airline.py
```

Keep the CSV in the same folder as the `.py` file.

## The code in three parts

### 1) Import the dataset and prepare it

```python
df = pd.read_csv('airline-passengers.csv')
data = df['Passengers'].values.reshape(-1, 1).astype('float32')
scaler = MinMaxScaler()
data = scaler.fit_transform(data)
```

- `pd.read_csv` loads the CSV. We only need the `Passengers` column.
- `reshape(-1, 1)` makes it a single column, which the scaler expects.
- `MinMaxScaler` squeezes the values into the range 0–1. Neural networks train better on small numbers.

**Making sequences (the look-back window):**

```python
look_back = 12
for i in range(len(data) - look_back):
    X.append(data[i:i + look_back])
    y.append(data[i + look_back])
```

- Each input `X` is **12 months in a row**, and its target `y` is **the month right after**.
- Example: months 1–12 → predict month 13; months 2–13 → predict month 14; and so on.
- 12 is used because the data has a yearly pattern (summer peaks every year).

**Train/test split:**

```python
split = int(len(X) * 0.8)
```

- First 80% of the timeline is for training, last 20% for testing.
- We **don't shuffle** time-series data: the model must learn from the past and predict the future.

### 2) Build and train the three models

```python
for name, Layer in [('RNN', layers.SimpleRNN), ('LSTM', layers.LSTM), ('GRU', layers.GRU)]:
    model = Sequential([
        layers.Input(shape=(look_back, 1)),
        Layer(50),
        layers.Dense(1)
    ])
```

- The loop builds the **same model three times**, swapping only the recurrent layer. This keeps the comparison fair.
- `Input(shape=(12, 1))`: 12 time steps, 1 value (passenger count) per step.
- `Layer(50)`: the recurrent layer with 50 units.
- `Dense(1)`: one output, the predicted passenger count. No activation, because it's a number, not a class.

```python
model.compile(optimizer='adam', loss='mse')
model.fit(X_train, y_train, epochs=100, batch_size=8, verbose=0)
```

- `loss='mse'` (mean squared error) is the standard loss for predicting numbers.
- Same `look_back`, `epochs` and `batch_size` for all three models, as the question asks.
- `verbose=0` hides the per-epoch log so the output stays clean.

**The three layers:**

| Model | Idea | Weakness / strength |
|---|---|---|
| **SimpleRNN** | Passes a hidden state from one time step to the next, so it "remembers" earlier steps | Suffers from the **vanishing gradient** problem, so it forgets long-ago steps |
| **LSTM** | Adds a **cell state** and **3 gates** (forget, input, output) that decide what to keep, add and output | Remembers long-term patterns well, but is the slowest (most weights) |
| **GRU** | Simplified LSTM with **2 gates** (update, reset) and no separate cell state | Almost as good as LSTM, faster and fewer weights |

### 3) Predict and calculate MSE and RMSE

```python
pred = scaler.inverse_transform(model.predict(X_test))
actual = scaler.inverse_transform(y_test)
mse = mean_squared_error(actual, pred)
print(name, "MSE:", mse, "RMSE:", np.sqrt(mse))
```

- `inverse_transform` converts predictions from the 0–1 scale back to real passenger numbers, so the error is meaningful.
- **MSE** = average of (actual − predicted)². Squaring punishes big errors more.
- **RMSE** = √MSE. It's in the **same units as the data** (thousands of passengers), so it's easier to interpret. **Lower is better.**

## Expected output (roughly)

LSTM and GRU usually get a lower RMSE than SimpleRNN. Exact numbers change on each run because the weights start randomly.

## Quick viva points

- **What is a time series?** Data recorded in time order, where earlier values help predict later ones.
- **Why not shuffle the data?** The order matters; shuffling would let the model "see the future".
- **What is the look-back window?** How many past time steps the model sees to make one prediction (here 12 months).
- **What is the vanishing gradient problem?** In long sequences, gradients shrink as they flow back through time, so a SimpleRNN stops learning from early steps. LSTM and GRU gates fix this.
- **LSTM gates:** forget gate (what to throw away), input gate (what new info to store), output gate (what to output).
- **GRU gates:** update gate (how much past to keep), reset gate (how much past to ignore).
- **Why RMSE over MSE for reporting?** RMSE is in the same units as the original data.
- **Which is best usually?** LSTM or GRU. GRU is often preferred when accuracy is similar because it trains faster.
