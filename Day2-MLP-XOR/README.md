# Day 2: Multi-Layer Perceptron (MLP) for XOR

## Question

Design and implement a Multi-Layer Perceptron using appropriate activation functions to solve non-linearly separable problems such as XOR. Analyze the effect of learning rate and epochs.

> **Note:** This is the minimal exam version. It trains the MLP on XOR with two learning rates and two epoch counts, and prints a confusion matrix for each. Plots and the written analysis are left out on purpose.

## How to run

```
pip install tensorflow scikit-learn
python mlp_xor.py
```

## The code in three parts

### 1) The dataset: XOR

```python
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([0, 1, 1, 0])
```

| x1 | x2 | XOR |
|---|---|---|
| 0 | 0 | 0 |
| 0 | 1 | 1 |
| 1 | 0 | 1 |
| 1 | 1 | 0 |

XOR is 1 when the inputs are **different**. Plot the 4 points and you'll see no single straight line can separate the 1s from the 0s, so XOR is **not linearly separable**. A single perceptron (Day 1) fails here.

The whole dataset is just these 4 rows, so we train and test on the same data.

### 2) Build and train the MLP

```python
model = Sequential([
    layers.Input(shape=(2,)),
    layers.Dense(8, activation='tanh'),
    layers.Dense(1, activation='sigmoid')
])
```

| Layer | What it does |
|---|---|
| `Input(2)` | The two inputs x1 and x2 |
| `Dense(8, tanh)` | **Hidden layer.** This is what makes it "multi-layer". The non-linear `tanh` activation lets the network bend the decision boundary |
| `Dense(1, sigmoid)` | Output between 0 and 1. Above 0.5 = class 1 |

**Why activation functions matter:** without a non-linear activation, stacking layers is still just one straight line. `tanh` (or `relu`) in the hidden layer is what lets the MLP solve XOR.

**Learning rate and epochs loop:**

```python
for lr in [0.01, 0.1]:
    for epochs in [100, 500]:
        ...
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr), ...)
        model.fit(X, y, epochs=epochs, verbose=0)
```

- Trains a **fresh model** for each of the 4 combinations, so they're compared fairly.
- **Learning rate:** how big a step the weights take on each update.
- **Epochs:** how many times the model goes through the data.
- `binary_crossentropy` is the loss for a two-class (0/1) output.

### 3) Print the confusion matrix

```python
y_pred = (model.predict(X, verbose=0) > 0.5).astype(int).ravel()
print(confusion_matrix(y, y_pred))
```

- `> 0.5` turns the sigmoid output into 0 or 1.
- A perfect result is `[[2 0] [0 2]]`: both 0s and both 1s correct.

## Expected output (roughly)

| Learning rate | Epochs | Usually |
|---|---|---|
| 0.01 | 100 | Not solved yet: too small a step, too few epochs |
| 0.01 | 500 | Often solved, or close |
| 0.1 | 100 | Usually solved |
| 0.1 | 500 | Solved |

Results can change between runs because the starting weights are random.

## Effect of learning rate and epochs (viva)

- **Learning rate too small:** learns very slowly; needs many more epochs.
- **Learning rate too large:** jumps around and may never settle on a good answer.
- **Too few epochs:** the model hasn't finished learning (underfitting).
- **More epochs:** more time to learn, but past a point it adds nothing (and on bigger datasets can cause overfitting).
- A higher learning rate needs fewer epochs; a lower one needs more.

## Quick viva points

- **Why can't a single perceptron solve XOR?** It can only draw one straight line, and XOR's classes can't be split by one line.
- **How does an MLP solve it?** The hidden layer transforms the inputs into a new space where the classes become linearly separable.
- **What is backpropagation?** The algorithm that calculates how much each weight caused the error, working backwards from the output, so the weights can be updated.
- **Common activation functions:** sigmoid (0 to 1), tanh (−1 to 1), ReLU (max(0, x)).
