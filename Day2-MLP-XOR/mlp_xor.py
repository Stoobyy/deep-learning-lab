import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Sequential
from sklearn.metrics import confusion_matrix

# 1) Dataset: XOR truth table
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([0, 1, 1, 0])

# 2) Train an MLP with different learning rates and epochs
for lr in [0.01, 0.1]:
    for epochs in [100, 500]:
        model = Sequential([
            layers.Input(shape=(2,)),
            layers.Dense(8, activation='tanh'),
            layers.Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                      loss='binary_crossentropy', metrics=['accuracy'])
        model.fit(X, y, epochs=epochs, verbose=0)

        # 3) Confusion matrix
        y_pred = (model.predict(X, verbose=0) > 0.5).astype(int).ravel()
        print("Learning rate:", lr, "Epochs:", epochs)
        print(confusion_matrix(y, y_pred))
