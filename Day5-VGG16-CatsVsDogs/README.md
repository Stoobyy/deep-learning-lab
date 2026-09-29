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
- Apply: Rotation, Zoom, Horizontal Flip, Rescaling
- Retrain model
- Compare: Accuracy, Loss, Training time
- Analyze effect of augmentation

> **Note:** This is the minimal exam version. It trains the model without augmentation, retrains it with rotation, zoom, horizontal flip and rescaling, and prints a confusion matrix for each. The accuracy/loss/time comparison and the written analysis are left out on purpose.

## Dataset setup

The code expects the images in this folder layout, next to the `.py` file:

```
dataset/
  train/
    cats/   (cat images)
    dogs/   (dog images)
  test/
    cats/
    dogs/
```

The folder names become the labels automatically, in alphabetical order: **cats = 0, dogs = 1**. If the lab gives you the dataset with different folder names, just change the paths in the two `image_dataset_from_directory` lines.

The Kaggle "Dogs vs. Cats" dataset works. A few hundred images per class is enough for the exam and keeps training fast.

## How to run

```
pip install tensorflow scikit-learn
python vgg16_cats_dogs.py
```

The VGG16 weights (about 60 MB) download automatically the first time.

## The code in three parts

### 1) Import the dataset

```python
train_ds = tf.keras.utils.image_dataset_from_directory('dataset/train', image_size=(150, 150), batch_size=32)
test_ds = tf.keras.utils.image_dataset_from_directory('dataset/test', image_size=(150, 150), batch_size=32, shuffle=False)
```

- `image_dataset_from_directory` reads images from folders and uses each **folder name as the class label**.
- `image_size=(150, 150)` resizes every image to 150×150 so they all match.
- `batch_size=32` feeds 32 images at a time.
- `shuffle=False` on the test set keeps the order fixed, so predictions line up with the true labels when we build the confusion matrix.

### 2) Build the model and train it

**Transfer learning with VGG16:**

```python
base = VGG16(weights='imagenet', include_top=False, input_shape=(150, 150, 3))
base.trainable = False
```

- **VGG16** is a 16-layer CNN already trained on ImageNet (1.4 million images, 1000 classes). It already knows how to detect edges, textures and shapes.
- `weights='imagenet'` loads those pre-trained weights.
- `include_top=False` removes VGG16's original 1000-class output layers, because we only need 2 classes.
- `base.trainable = False` **freezes** VGG16 so its weights don't change. Only our new layers are trained. This is **transfer learning**: reusing knowledge from one task for another.

**Data augmentation layers** (only added when `augment=True`):

| Layer | What it does |
|---|---|
| `RandomRotation(0.1)` | Rotates each image randomly by up to ±10% of a full turn (±36°) |
| `RandomZoom(0.1)` | Zooms in or out randomly by up to 10% |
| `RandomFlip('horizontal')` | Randomly mirrors the image left–right |
| `Rescaling(1./255)` | Scales pixels from 0–255 to 0–1 (used in both runs) |

Augmentation creates slightly different versions of each training image every epoch, so the model sees more variety and **overfits less**. These random layers are only active during training; they do nothing during `predict`.

**Our classifier on top:**

| Layer | What it does |
|---|---|
| `Flatten()` | Turns VGG16's output (4 × 4 × 512) into one row |
| `Dense(128, relu)` | Learns to combine VGG16's features for cats vs dogs |
| `Dense(1, sigmoid)` | One output between 0 and 1. Close to 0 = cat, close to 1 = dog |

**Training:**

```python
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(train_ds, epochs=3)
```

- Only **2 classes**, so we use **sigmoid + binary_crossentropy** (not softmax + categorical, like Days 3 and 4).
- The `for augment in [False, True]` loop trains a fresh model without augmentation, then retrains a fresh one with it.

### 3) Print the confusion matrix

```python
y_true = np.concatenate([y for x, y in test_ds])
y_pred = (model.predict(test_ds) > 0.5).astype(int).ravel()
print(confusion_matrix(y_true, y_pred))
```

- `y_true` collects the real labels from every test batch into one array.
- `> 0.5` turns each sigmoid probability into a class: above 0.5 means dog (1), otherwise cat (0).
- `.ravel()` flattens the predictions into a 1D array.
- The result is a 2×2 table:

|  | Predicted cat | Predicted dog |
|---|---|---|
| **Actual cat** | correct cats | cats called dogs |
| **Actual dog** | dogs called cats | correct dogs |

## Expected output (roughly)

Both runs usually reach about 85–95% test accuracy, depending on how many images you use. The augmented model often trains a bit slower and may score slightly lower on training accuracy, but it tends to generalise better to new images.

## Quick viva points

- **What is transfer learning?** Taking a model trained on a big dataset (ImageNet) and reusing it for a new task, training only the new top layers.
- **Why freeze VGG16?** Its features are already good, we have little data, and training only the top is much faster.
- **What does `include_top=False` do?** Removes VGG16's original classifier so we can add our own for 2 classes.
- **Why data augmentation?** More variety in training images, so less overfitting and better accuracy on new images.
- **Why sigmoid, not softmax?** There are only 2 classes, so one output (the probability of dog) is enough.
- **Why is VGG called VGG16?** It has 16 layers with weights: 13 convolution layers and 3 fully connected layers.
