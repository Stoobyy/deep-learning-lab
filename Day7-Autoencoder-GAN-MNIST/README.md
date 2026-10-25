# Day 7: Autoencoder and GAN on MNIST

## Question

Design and implement Autoencoders and Generative Adversarial Networks (GANs) for representation learning and synthetic data generation.

**Dataset:** MNIST

**7.1 Part A: Autoencoder**
- Build encoder
- Build decoder
- Reconstruct handwritten digit images
- Compare: original image, reconstructed image

**7.2 Part B: GAN**
- Build generator
- Build discriminator
- Generate synthetic handwritten digits
- Visualize generated samples
- Analyze GAN performance

> **Note:** This is the minimal exam version. Part A builds the encoder and decoder, reconstructs the test images and compares them with the originals using a number (MSE). Part B builds the generator and discriminator, trains them, and generates 10 new digits. The image plots and the written analysis are left out on purpose.
>
> **Why no confusion matrix?** Neither model classifies anything. The autoencoder rebuilds images and the GAN creates new ones, so there are no predicted classes to count. Reconstruction error (Part A) and the two losses (Part B) are used instead.

## Files

- `autoencoder.py`: Part A
- `gan.py`: Part B

## How to run

```
pip install tensorflow scikit-learn
python autoencoder.py
python gan.py
```

MNIST downloads automatically the first time.

---

## Part A: Autoencoder

**Idea:** squeeze an image down to a small code, then rebuild the image from that code. If the rebuilt image looks like the original, the small code has captured the important information. That small code is the **learned representation**.

```
image (784) → ENCODER → code (32) → DECODER → rebuilt image (784)
```

### 1) Import the dataset

```python
(x_train, _), (x_test, _) = tf.keras.datasets.mnist.load_data()
x_train = x_train.reshape(-1, 784) / 255.0
```

- The `_` throws away the labels. An autoencoder is **unsupervised**: the image is both the input and the target.
- Images are flattened to 784 values and scaled to 0–1, as on Day 3.

### 2) Build the encoder and decoder, then train

| Part | Layers | What it does |
|---|---|---|
| **Encoder** | 784 → 128 → 32 | Compresses the image into 32 numbers (the **bottleneck** / latent code) |
| **Decoder** | 32 → 128 → 784 | Rebuilds the full image from those 32 numbers |

- The decoder's last layer uses **sigmoid** so every output pixel is between 0 and 1, like the input.
- `Sequential([encoder, decoder])` joins them into one model.

```python
autoencoder.fit(x_train, x_train, ...)
```

- Notice `x_train` is passed **twice**: input and target are the same image. The model learns to copy its input through the small bottleneck.
- `binary_crossentropy` works as the loss because pixels are between 0 and 1.

### 3) Reconstruct and compare

```python
reconstructed = autoencoder.predict(x_test)
print("Reconstruction MSE:", mean_squared_error(x_test, reconstructed))
```

- `predict` passes each test image through encoder and decoder.
- **MSE** compares every original pixel with its rebuilt pixel. **Closer to 0 means the reconstruction is closer to the original.** Expect a small value, roughly 0.01–0.02.

---

## Part B: GAN

**Idea:** two networks compete.

- The **generator** turns random noise into a fake image.
- The **discriminator** looks at an image and says real (1) or fake (0).
- The generator gets better at fooling the discriminator, and the discriminator gets better at catching fakes. Over time the fakes start to look like real digits.

A common analogy: the generator is a forger making fake notes, and the discriminator is the police checking them.

### 1) Import the dataset

Same as Part A, but only the training images are needed.

### 2) Build the generator and discriminator

| Network | Layers | Output |
|---|---|---|
| **Generator** | 100 (noise) → 128 → 784 | A fake image. Sigmoid keeps pixels in 0–1 |
| **Discriminator** | 784 → 128 → 1 | Probability the image is real. Sigmoid output |

**The combined GAN:**

```python
discriminator.trainable = False
gan = Sequential([generator, discriminator])
```

- Noise → generator → fake image → discriminator → "real or fake?"
- The discriminator is **frozen** inside the GAN, so training the GAN only updates the **generator**.

**One training step:**

| Step | What happens | Labels |
|---|---|---|
| 1 | Generator makes 64 fake images from random noise | – |
| 2 | Pick 64 real images | – |
| 3 | Train discriminator on real + fake | real = 1, fake = 0 |
| 4 | Train the GAN on noise, telling it the fakes are **real** | 1 |

- Step 4 is the trick: we label the fakes as real, so the generator is pushed to make images the discriminator would call real.
- `discriminator.trainable = True / False` is switched before each step so the right network gets updated.
- Losses are printed every 50 steps, so you can see it's still running. The full 2000 steps take a few minutes.

### 3) Generate synthetic digits

```python
samples = generator.predict(np.random.normal(0, 1, (10, 100)))
```

- 10 random noise vectors → 10 brand-new handwritten digits that aren't in the dataset.
- Reshaped to `(10, 28, 28)`: 10 images of 28×28.

### Reading the losses

- **D loss** high, **G loss** low → the generator is winning (fooling the discriminator).
- **D loss** near 0, **G loss** very high → the discriminator is winning and the generator isn't learning.
- In a healthy GAN, both stay in a similar range and go up and down as they compete. They don't steadily fall to 0 like a normal model's loss.

---

## Quick viva points

- **Autoencoder vs GAN?** An autoencoder learns to **compress and rebuild** existing images. A GAN learns to **create new** images from random noise.
- **What is the bottleneck / latent space?** The small middle layer (32 numbers) that holds the compressed representation.
- **Why is an autoencoder unsupervised?** It needs no labels; the image is its own target.
- **Uses of autoencoders:** dimensionality reduction, denoising images, anomaly detection.
- **Why freeze the discriminator inside the GAN?** So that training the GAN only improves the generator.
- **What is mode collapse?** When the generator keeps making the same few images instead of a variety, because those few fool the discriminator.
- **Why is GAN training hard?** The two networks have to stay balanced; if one becomes much stronger, the other stops learning.
