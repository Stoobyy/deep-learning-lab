
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.applications import VGG16
from sklearn.metrics import confusion_matrix

# Dataset
data_dir = "dataset"

train_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    image_size=(150, 150),
    class_names=["cats", "dogs"],
    validation_split=0.2,
    subset="training",
    seed=42
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    image_size=(150, 150),
    class_names=["cats", "dogs"],
    validation_split=0.2,
    subset="validation",
    seed=42,
    shuffle=False
)

# VGG16 Model
def build_model(augment):
    base = VGG16(weights="imagenet", include_top=False,
                 input_shape=(150, 150, 3))
    base.trainable = False

    model = tf.keras.Sequential()
    model.add(layers.Input(shape=(150, 150, 3))) #3 because color image, if images were loaded with color_mode='grayscale', it wouldve been 1

    if augment:
        model.add(layers.RandomRotation(0.1))
        model.add(layers.RandomZoom(0.1))
        model.add(layers.RandomFlip("horizontal"))

    model.add(layers.Rescaling(1./255))
    model.add(layers.Rescaling(255.0))
    model.add(layers.Lambda(tf.keras.applications.vgg16.preprocess_input))
    model.add(base)
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation="relu"))
    model.add(layers.Dense(2, activation="softmax"))

    return model

# Train without and with augmentation
for augment in [False, True]:
    model = build_model(augment)

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    model.fit(train_ds, epochs=3, validation_data=val_ds)

    loss, accuracy = model.evaluate(val_ds)

    y_true = np.concatenate([y.numpy() for x, y in val_ds])
    y_pred = np.argmax(model.predict(val_ds), axis=1)

    print("Augmentation:", augment)
    print("Accuracy:", accuracy)
    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))
