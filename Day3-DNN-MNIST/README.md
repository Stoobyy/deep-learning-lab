# Day 3: Deep Feed Forward Neural Network (DNN) on MNIST

## Question

Design and implement a Deep Feed Forward Neural Network (DNN) for classification. Train the model using different optimization algorithms and regularization techniques.

**Dataset:** MNIST Handwritten Digits
**Classes used:** Digits 0–9

**Network architecture**
- Input layer: 784 neurons
- Hidden layer 1: 128 neurons
- Hidden layer 2: 64 neurons
- Output layer: 10 neurons (Softmax)

**Tasks**
- Train using: SGD (Gradient Descent), Adam, RMSProp
- Apply: Dropout, L1 Regularization, L2 Regularization, Early Stopping
- Plot: Training Accuracy, Validation Accuracy, Training Loss, Validation Loss
- Compare optimizer performance

> **Note:** This is the minimal exam version. It covers the architecture, the three optimizers, all four regularization techniques, and prints a confusion matrix for each optimizer. The plots and the written comparison are left out on purpose.

## How to run

```
pip install tensorflow scikit-learn
python dnn_mnist.py
```

MNIST downloads automatically the first time, so no dataset file is needed.

## The code in three parts

### 1) Import the dataset

```python
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = x_train.reshape(-1, 784) / 255.0
x_test = x_test.reshape(-1, 784) / 255.0
```

- MNIST has 60,000 training images and 10,000 test images. Each image is a 28×28 grayscale handwritten digit.
- `reshape(-1, 784)` flattens each 28×28 image into one row of 784 numbers (28 × 28 = 784). This is why the input layer has 784 neurons. The `-1` tells numpy to work out the number of rows itself.
- `/ 255.0` scales pixel values from 0–255 down to 0–1. Small inputs make training faster and more stable.

### 2) Build the model and train it

```python
layers.Input(shape=(784,)),
layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l1(0.0001)),
layers.Dropout(0.2),
layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
layers.Dense(10, activation='softmax')
```

| Line | What it does |
|---|---|
| `Input(shape=(784,))` | Input layer: 784 pixel values |
| `Dense(128, activation='relu')` | Hidden layer 1. Every neuron connects to every input. ReLU outputs `max(0, x)` |
| `Dense(64, activation='relu')` | Hidden layer 2 |
| `Dense(10, activation='softmax')` | Output layer. One neuron per digit (0–9). Softmax turns the outputs into probabilities that add up to 1 |

**Regularization** (stops the model memorising the training data, i.e. overfitting):

| Technique | Where in the code | Idea |
|---|---|---|
| **L1** | `regularizers.l1(0.0001)` on layer 1 | Adds the sum of absolute weights to the loss. Pushes many weights to exactly 0 |
| **L2** | `regularizers.l2(0.001)` on layer 2 | Adds the sum of squared weights to the loss. Keeps all weights small |
| **Dropout** | `Dropout(0.2)` | Randomly switches off 20% of neurons each training step, so the network can't rely on any single neuron |
| **Early stopping** | `EarlyStopping(monitor='val_loss', patience=2)` | Stops training if validation loss hasn't improved for 2 epochs in a row |

**Training loop:**

```python
for opt in ['sgd', 'adam', 'rmsprop']:
    model = build_model()
    model.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=5, validation_split=0.1, callbacks=[early])
```

- A **fresh model** is built for each optimizer so the comparison is fair.
- `sparse_categorical_crossentropy` is the loss for multi-class problems where labels are plain integers (0–9). It's "sparse" because the labels aren't one-hot encoded.
- `validation_split=0.1` holds back 10% of the training data to check the model on unseen data during training. Early stopping watches this.
- `epochs=5` means the model goes through the full training set up to 5 times.

**The three optimizers** (they decide how the weights are updated):

| Optimizer | Idea |
|---|---|
| **SGD** | Plain gradient descent: weight = weight − learning_rate × gradient. Simple but slow |
| **RMSProp** | Adapts the learning rate for each weight using a running average of recent squared gradients |
| **Adam** | Combines momentum with RMSProp's adaptive learning rate. Usually the fastest and most accurate |

### 3) Print the confusion matrix

```python
y_pred = model.predict(x_test).argmax(axis=1)
print(confusion_matrix(y_test, y_pred))
```

- `model.predict` gives 10 probabilities per image. `argmax(axis=1)` picks the digit with the highest probability.
- The confusion matrix is a 10×10 table. **Rows = actual digit, columns = predicted digit.**
- The **diagonal** counts correct predictions. Anything off the diagonal is a mistake. For example, the value at row 4, column 9 is how many 4s were predicted as 9s.

## Expected output (roughly)

Adam and RMSProp usually reach about 97% test accuracy in 5 epochs. SGD is slower and lower, about 92–94%. Exact numbers change on each run.

## Quick viva points

- **Why 784 inputs?** 28 × 28 pixels, flattened.
- **Why softmax at the output?** It gives a probability for each of the 10 classes.
- **Why ReLU in hidden layers?** It's simple, fast, and avoids the vanishing gradient problem.
- **L1 vs L2?** L1 makes weights exactly zero (sparse). L2 makes them small but not zero.
- **What is overfitting?** Doing well on training data but poorly on new data. Dropout, L1, L2 and early stopping all reduce it.
