import tensorflow as tf
from tensorflow.keras import layers
from sklearn.metrics import confusion_matrix

# 1) Import dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = x_train.reshape(-1, 28, 28, 1) / 255.0
x_test = x_test.reshape(-1, 28, 28, 1) / 255.0

# 2) Build CNN: Conv -> Pool -> Conv -> Pool -> Dense -> Softmax
model = tf.keras.Sequential([
    layers.Input(shape=(28, 28, 1)),
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=5, validation_split=0.1)

# Evaluate accuracy
loss, acc = model.evaluate(x_test, y_test)
print("Test accuracy:", acc)

# 3) Confusion matrix
y_pred = model.predict(x_test).argmax(axis=1)
print(confusion_matrix(y_test, y_pred))
