import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.applications import VGG16
from sklearn.metrics import confusion_matrix

# 1) Import dataset (folders: dataset/train/cats, dataset/train/dogs, same for test)
train_ds = tf.keras.utils.image_dataset_from_directory('dataset/train', image_size=(150, 150), batch_size=32)
test_ds = tf.keras.utils.image_dataset_from_directory('dataset/test', image_size=(150, 150), batch_size=32, shuffle=False)

# 2) Build model: pre-trained VGG16 + our own classifier on top
def build_model(augment):
    base = VGG16(weights='imagenet', include_top=False, input_shape=(150, 150, 3))
    base.trainable = False   # transfer learning: freeze VGG16

    model = tf.keras.Sequential()
    model.add(layers.Input(shape=(150, 150, 3)))
    if augment:
        model.add(layers.RandomRotation(0.1))
        model.add(layers.RandomZoom(0.1))
        model.add(layers.RandomFlip('horizontal'))
    model.add(layers.Rescaling(1./255))
    model.add(base)
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))
    return model

y_true = np.concatenate([y for x, y in test_ds])

# Train without augmentation, then retrain with augmentation
for augment in [False, True]:
    model = build_model(augment)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(train_ds, epochs=3)

    # 3) Confusion matrix
    y_pred = (model.predict(test_ds) > 0.5).astype(int).ravel()
    print("Augmentation:", augment)
    print(confusion_matrix(y_true, y_pred))
