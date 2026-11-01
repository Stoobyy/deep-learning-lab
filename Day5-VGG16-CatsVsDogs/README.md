# Day 5: VGG16 Transfer Learning on Cats vs Dogs (with Data Augmentation)

## Question

Design and implement VGG16 architecture for image classification. Apply data augmentation and analyze performance.

**Dataset:** Cats vs Dogs  
**Classes used:** Cat, Dog

**Model**
- Pre-trained VGG16
- Transfer learning

**Tasks**
- Train without augmentation
- Apply rotation, zoom, horizontal flip and rescaling
- Retrain with augmentation
- Evaluate accuracy, loss and confusion matrix
- Analyze the effect of augmentation

> **Note:** This is the simple exam version. The script trains two separate models (without and with augmentation) and prints validation accuracy and a confusion matrix for each. It does not measure training time or automatically calculate a comparison table.

## Dataset setup

Place the images in these folders, relative to where you run the script:

```text
dataset/
  cats/
    cat1.jpg
    ...
  dogs/
    dog1.jpg
    ...
```

The script automatically splits the images into **80% training** and **20% validation** using `validation_split=0.2` and `seed=42`. You do not need separate `train` and `test` folders.

The labels are **cats = 0** and **dogs = 1**, specified by `class_names=["cats", "dogs"]`. Images are resized to **150 × 150**.

## How to run

```bash
pip install numpy tensorflow scikit-learn
python vgg16_cats_dogs.py
```

The pre-trained ImageNet VGG16 weights are downloaded automatically on the first run if they are not already cached.

## The code in three parts

### 1) Import the dataset

```python
data_dir = r"dataset"

train_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir, image_size=(150, 150), class_names=["cats", "dogs"],
    validation_split=0.2, subset="training", seed=42
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir, image_size=(150, 150), class_names=["cats", "dogs"],
    validation_split=0.2, subset="validation", seed=42, shuffle=False
)
```

- `image_dataset_from_directory` loads and labels the images from their folders.
- `validation_split=0.2` reserves 20% for validation.
- `seed=42` makes the split reproducible.
- `shuffle=False` keeps validation image order fixed for the confusion matrix.
- The default batch size is **32**.

### 2) Build and train VGG16

**Transfer learning:**

```python
base = VGG16(weights="imagenet", include_top=False,
             input_shape=(150, 150, 3))
base.trainable = False
```

- `weights="imagenet"` loads pre-trained VGG16 features.
- `include_top=False` removes its original classification head.
- `base.trainable=False` freezes the pre-trained layers, so only the new classifier is trained.

**Augmentation** (added only when `augment=True`):

| Layer | Purpose |
|---|---|
| `RandomRotation(0.1)` | Random rotation up to ±36° |
| `RandomZoom(0.1)` | Random zoom of up to 10% |
| `RandomFlip("horizontal")` | Random left-right flip |

**Preprocessing** (used in both models):

```python
model.add(layers.Rescaling(1./255))
model.add(layers.Rescaling(255.0))
model.add(layers.Lambda(tf.keras.applications.vgg16.preprocess_input))
```

The two rescaling operations cancel each other out. Rescaling is done explicitly because its mentioned in the question, otherwise you can skip both rescaling layers
`preprocess_input` then prepares the original-range pixel values as expected by pre-trained VGG16. Thus, rescaling is present in the code but does not change the final input range on its own.

**Classifier:**

| Layer | Purpose |
|---|---|
| `Flatten()` | Converts VGG16 feature maps into a vector |
| `Dense(128, activation="relu")` | Learns features for cat/dog classification |
| `Dense(2, activation="softmax")` | Outputs probabilities for cat (0) and dog (1) |

**Training:**

```python
for augment in [False, True]:
    model = build_model(augment)
    model.compile(optimizer="adam",
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    model.fit(train_ds, epochs=3, validation_data=val_ds)
```

The loop builds a **fresh model** for each run: first without augmentation, then with augmentation. The loss is `sparse_categorical_crossentropy` because labels are integer class IDs (`0` and `1`) and the output has two softmax units.

### 3) Evaluate and print the confusion matrix

```python
loss, accuracy = model.evaluate(val_ds)
y_true = np.concatenate([y.numpy() for x, y in val_ds])
y_pred = np.argmax(model.predict(val_ds), axis=1)

print("Augmentation:", augment)
print("Accuracy:", accuracy)
print("Confusion Matrix:")
print(confusion_matrix(y_true, y_pred))
```

- `evaluate` calculates validation loss and accuracy; the script prints **accuracy**, not the loss value.
- `argmax(..., axis=1)` chooses the class with the highest softmax probability.
- The confusion matrix is a 2 × 2 table:

| | Predicted cat | Predicted dog |
|---|---|---|
| **Actual cat** | Correct cats | Cats predicted as dogs |
| **Actual dog** | Dogs predicted as cats | Correct dogs |

## Performance analysis

Compare the accuracy and confusion matrices printed for `Augmentation: False` and `Augmentation: True`.

- **Without augmentation:** The model learns from the original training images.
- **With augmentation:** The model sees randomly rotated, zoomed and flipped images during training, which can help reduce overfitting.
- **Results vary:** Augmentation does not guarantee higher validation accuracy, especially with only 3 epochs.
- **Loss:** Training and validation loss appear in the `fit` logs; validation loss is also calculated by `evaluate`, but not separately printed.
- **Training time:** Not measured by this script.

## Quick viva points

- **What is transfer learning?** Reusing features learned by a pre-trained model for a new task.
- **Why freeze VGG16?** To train only the new classifier and reduce training cost.
- **What does `include_top=False` do?** Removes VGG16's original classifier.
- **Why use data augmentation?** To add variation to training images and potentially reduce overfitting.
- **Why softmax?** It outputs probabilities for both classes.
- **Why sparse categorical crossentropy?** The labels are integers, not one-hot vectors.
- **Why is it called VGG16?** It has 16 weight-bearing layers: 13 convolutional and 3 fully connected layers in its original architecture.
