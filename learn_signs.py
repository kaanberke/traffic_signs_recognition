import cv2
import numpy as np
import os
import sys
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import questionary

EPOCHS = 500
BATCH = 128
IMG_WIDTH = 30
IMG_HEIGHT = 30
IMG_CHANNEL = 3
NUM_CATEGORIES = 43


def main():
    # Check command-line arguments
    if len(sys.argv) not in [2]:
        sys.exit("Usage: python learn_signs.py data_directory")

    # Get image arrays and labels for all image files
    images, labels = load_data(sys.argv[1])

    # Split data into training and testing sets
    labels = tf.keras.utils.to_categorical(labels, num_classes=NUM_CATEGORIES)

    x_train, x_val, y_train, y_val = train_test_split(
        np.array(images), np.array(labels), test_size=0.3
    )

    x_test, x_val, y_test, y_val = train_test_split(
        x_val, y_val, test_size=0.33
    )

    # Get a compiled neural network
    model = get_model()

    # Create a folder to save model checkpoints.
    if not os.path.exists("models"):
        os.makedirs("models")

    model_filepath = os.path.join("models", "model.{epoch:02d}-{val_loss:.4f}.h5")
    cb = [
        tf.keras.callbacks.EarlyStopping(patience=20),
        tf.keras.callbacks.ModelCheckpoint(filepath=model_filepath,
                                           save_best_only=True),
        tf.keras.callbacks.TensorBoard(log_dir="logs")
    ]

    # Fit model on training data
    history = model.fit(x_train, y_train,
                        validation_data=(x_val, y_val),
                        batch_size=BATCH,
                        epochs=EPOCHS,
                        callbacks=cb,
                        verbose=1,
                        use_multiprocessing=True,
                        workers=-1)

    fig, ax = plt.subplots(1, 2)

    fig.set_size_inches(22.5, 10.5, forward=True)

    ax[0].plot(history.history.get("loss"), label="loss", c="blue")
    ax[0].plot(history.history.get("val_loss"), label="val_loss", c="green")
    ax[0].legend()

    ax[1].plot(history.history.get("acc"), label="acc", c="orange")
    ax[1].plot(history.history.get("val_acc"), label="val_acc", c="red")
    ax[1].legend()

    plt.show()
    if not os.path.exists("plots"):
        os.makedirs("plots")
    plt.savefig(os.path.join("plots", "loss-vall_loss and acc-val_acc.jpg"))


    # Evaluate neural network performance
    model.evaluate(x_test, y_test, verbose=2)

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

    models_paths = os.listdir("models")
    models_paths.insert(0, "Train from scratch")
    if len(models_paths) != 0:
        answer = questionary.select(
            "Which old model you want to load?",
            choices=models_paths
                           ).ask()
        if answer != "Train from scratch":
            model = tf.keras.models.load_model(os.path.join("models", answer))
            return model

    def default_conv(filters, kernel_size=(3, 3), strides=1, padding="SAME", **kwargs):
        return [
            tf.keras.layers.Conv2D(filters, kernel_size, strides=strides, padding=padding, **kwargs),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation(tf.keras.activations.relu),
            tf.keras.layers.Dropout(0.5),
        ]

    layers = [
        tf.keras.layers.Input(shape=(30, 30, 3)),

        *default_conv(512, (7, 7), strides=2, padding="SAME"),
        *default_conv(256, (3, 3), strides=1, padding="SAME"),
        *default_conv(128, (3, 3), strides=1, padding="SAME"),

        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation="relu"),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(43, activation="softmax"),

        # FCNN
        # tf.keras.layers.Conv2D(43, (15, 15), activation="softmax"),
        # tf.keras.layers.Flatten()
    ]
    model = tf.keras.models.Sequential(layers=layers)
    model.compile(optimizer="Adam", loss="categorical_crossentropy", metrics=["acc"])
    model.summary()

    return model


if __name__ == "__main__":
    main()
