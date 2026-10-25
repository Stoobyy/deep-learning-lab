import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Sequential

# 1) Import dataset (labels not needed)
(x_train, _), (_, _) = tf.keras.datasets.mnist.load_data()
x_train = x_train.reshape(-1, 784) / 255.0

# 2) Build generator: random noise (100) -> fake image (784)
generator = Sequential([
    layers.Input(shape=(100,)),
    layers.Dense(128, activation='relu'),
    layers.Dense(784, activation='sigmoid')
])

# Build discriminator: image (784) -> real (1) or fake (0)
discriminator = Sequential([
    layers.Input(shape=(784,)),
    layers.Dense(128, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])
discriminator.compile(optimizer='adam', loss='binary_crossentropy')

# GAN = generator followed by a frozen discriminator
discriminator.trainable = False
gan = Sequential([generator, discriminator])
gan.compile(optimizer='adam', loss='binary_crossentropy')

# Train
batch = 64
for step in range(500):
    noise = np.random.normal(0, 1, (batch, 100))
    fake = generator.predict(noise, verbose=0)
    real = x_train[np.random.randint(0, len(x_train), batch)]

    # Train discriminator: real = 1, fake = 0
    discriminator.trainable = True
    d_loss = discriminator.train_on_batch(np.concatenate([real, fake]),
                                          np.concatenate([np.ones((batch, 1)), np.zeros((batch, 1))]))

    # Train generator: try to make the discriminator say 1 (real)
    discriminator.trainable = False
    g_loss = gan.train_on_batch(noise, np.ones((batch, 1)))

    if step % 50 == 0:
        print("Step", step, "D loss:", d_loss, "G loss:", g_loss, flush=True)

# 3) Generate synthetic handwritten digits
samples = generator.predict(np.random.normal(0, 1, (10, 100)))
print("Generated images shape:", samples.reshape(-1, 28, 28).shape)
