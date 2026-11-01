import tensorflow as tf
from tensorflow.keras import layers

# Change this to the folder containing cats/ and dogs/.
data_dir = r"dataset"

# 80% training, 20% validation. Keep the same split and seed.
train_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    image_size=(28, 28),
    color_mode="grayscale",
    class_names=["cats", "dogs"],
    validation_split=0.2,
    subset="training",
    seed=42
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    image_size=(28, 28),
    color_mode="grayscale",
    class_names=["cats", "dogs"],
    validation_split=0.2,
    subset="validation",
    seed=42
)

model = tf.keras.Sequential([
    layers.Input(shape=(28, 28, 1)),
    layers.Rescaling(1./255),
    layers.Conv2D(32, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(64, activation="relu"),
    layers.Dense(2, activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)
model.fit(train_ds, epochs=5, validation_data=val_ds)
loss, accuracy = model.evaluate(val_ds)
print("Validation accuracy:", accuracy)

