import cv2
import numpy as np
import os
import sys
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


EPOCHS = 10
BATCH = 128
IMG_WIDTH = 30
IMG_HEIGHT = 30
IMG_CHANNEL = 3
NUM_CATEGORIES = 43
TEST_SIZE = 0.4


def main():
    # Check command-line arguments
    if len(sys.argv) not in [2, 3]:
        sys.exit("Usage: python traffic.py data_directory [model.h5]")

    # Get image arrays and labels for all image files
    images, labels = load_data(sys.argv[1])

    # Split data into training and testing sets
    labels = tf.keras.utils.to_categorical(labels, num_classes=NUM_CATEGORIES)
    x_train, x_test, y_train, y_test = train_test_split(
        np.array(images), np.array(labels), test_size=TEST_SIZE
    )

    # Get a compiled neural network
    model = get_model()

    cb = None
    """
    # Uncomment for using callbacks.
    # Create a folder to save model checkpoints.
    if not os.path.exists("models"):
        os.makedirs("models")

    model_filepath = os.path.join("models", "model.{epoch:02d}-{val_loss:.4f}.h5")
    cb = [
        # tf.keras.callbacks.EarlyStopping(patience=EPOCHS//3),
        tf.keras.callbacks.ModelCheckpoint(filepath=model_filepath,
                                           save_best_only=True),
        tf.keras.callbacks.TensorBoard(log_dir="logs")
    ]
    """
    # Fit model on training data
    history = model.fit(x_train, y_train,
                        validation_data=(x_test, y_test),
                        batch_size=BATCH,
                        epochs=EPOCHS,
                        callbacks=cb,
                        verbose=1,
                        use_multiprocessing=True,
                        workers=-1)

    # Evaluate neural network performance
    model.evaluate(x_test, y_test, verbose=2)

    # Save model to file
    if len(sys.argv) == 3:
        filename = sys.argv[2]
        model.save(filename)
        print(f"Model saved to {filename}.")

    return history


def load_data(data_dir):
    """
    Load image data from directory `data_dir`.

    Assume `data_dir` has one directory named after each category, numbered
    0 through NUM_CATEGORIES - 1. Inside each category directory will be some
    number of image files.

    Return tuple `(images, labels)`. `images` should be a list of all
    of the images in the data directory, where each image is formatted as a
    numpy ndarray with dimensions IMG_WIDTH x IMG_HEIGHT x 3. `labels` should
    be a list of integer labels, representing the categories for each of the
    corresponding `images`.
    """
    images = []
    labels = []

    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if not file.endswith(".ppm"):
                continue
            img = cv2.imread(os.path.join(root, file), 1)
            img = cv2.resize(img, dsize=(30, 30))
            images.append(img / 255.)
            labels.append(os.path.basename(root))

    return images, labels


def get_model():
    """
    Returns a compiled convolutional neural network model. Assume that the
    `input_shape` of the first layer is `(IMG_WIDTH, IMG_HEIGHT, 3)`.
    The output layer should have `NUM_CATEGORIES` units, one for each category.
    """

    layers = [
        tf.keras.layers.Input(shape=(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNEL)),

        tf.keras.layers.Conv2D(256, (3, 3), activation="relu"),
        tf.keras.layers.Dropout(0.5),

        tf.keras.layers.Conv2D(128, (3, 3), activation="relu"),
        tf.keras.layers.Dropout(0.5),

        tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
        tf.keras.layers.Dropout(0.5),

        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(164, activation="relu"),
        tf.keras.layers.Dense(NUM_CATEGORIES, activation="softmax"),
    ]
    model = tf.keras.models.Sequential(layers=layers)
    model.compile(optimizer="Adam", loss="categorical_crossentropy", metrics=["acc"])
    model.summary()
    return model


if __name__ == "__main__":
    main()
