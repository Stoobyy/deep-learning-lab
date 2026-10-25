import tensorflow as tf
from tensorflow.keras import layers, Sequential
from sklearn.metrics import mean_squared_error

# 1) Import dataset (labels not needed)
(x_train, _), (x_test, _) = tf.keras.datasets.mnist.load_data()
x_train = x_train.reshape(-1, 784) / 255.0
x_test = x_test.reshape(-1, 784) / 255.0

# 2) Build encoder: 784 -> 128 -> 32
encoder = Sequential([
    layers.Input(shape=(784,)),
    layers.Dense(128, activation='relu'),
    layers.Dense(32, activation='relu')
])

# Build decoder: 32 -> 128 -> 784
decoder = Sequential([
    layers.Input(shape=(32,)),
    layers.Dense(128, activation='relu'),
    layers.Dense(784, activation='sigmoid')
])

autoencoder = Sequential([encoder, decoder])
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
autoencoder.fit(x_train, x_train, epochs=5, batch_size=256, validation_data=(x_test, x_test))

# 3) Reconstruct test images and compare with the originals
reconstructed = autoencoder.predict(x_test)
print("Reconstruction MSE:", mean_squared_error(x_test, reconstructed))
