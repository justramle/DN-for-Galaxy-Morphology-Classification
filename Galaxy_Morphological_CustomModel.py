"""Train a custom CNN (Keras/TensorFlow) on the Galaxy10 DECals dataset."""

import os
from collections import Counter

import cv2
import numpy as np
from astroNN.datasets import galaxy10
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPool2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical

from plotting import plot_accuracy_and_loss_combined, plot_confusion

FEATURES = {
    0: "Distributed Galaxies",
    1: "Merging Galaxies",
    2: "Round Smooth Galaxies",
    3: "In-between Round Smooth Galaxies",
    4: "Cigar Shaped Smooth Galaxies",
    5: "Barred Spiral Galaxies",
    6: "Unbarred Tight Spiral Galaxies",
    7: "Unbarred Loose Spiral Galaxies",
    8: "Edge-on Galaxies without Bulge",
    9: "Edge-on Galaxies with Bulge",
}

IMG_SIZE = 128
MINORITY_CLASS = 4
MINORITY_MULTIPLIER = 3
BATCH_SIZE = 32
EPOCHS = 50
PATIENCE = 5

CHECKPOINT_DIR = "checkpoints"
MODEL_JSON_PATH = os.path.join(CHECKPOINT_DIR, "galaxy_model.json")
MODEL_WEIGHTS_PATH = os.path.join(CHECKPOINT_DIR, "best_model.h5")


def print_class_distribution(labels, title):
    counts = Counter(labels)
    print(title)
    for cls, count in sorted(counts.items()):
        print(f"  Class {cls}: {count} images")


def load_data():
    images, labels = galaxy10.load_data()
    labels = labels.astype(np.int64)
    images = images.astype(np.float32)
    print_class_distribution(labels, "Class distribution before augmentation:")
    return images, labels


def oversample_minority_class(images, labels, cls=MINORITY_CLASS, multiplier=MINORITY_MULTIPLIER):
    class_indices = np.where(labels == cls)[0]
    class_images = images[class_indices]
    target_count = len(class_images) * multiplier

    datagen = ImageDataGenerator(
        rotation_range=90,
        height_shift_range=0.2,
        width_shift_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
    )
    datagen.fit(class_images)

    generated_images = []
    for batch, _ in datagen.flow(class_images, labels[class_indices], batch_size=1):
        generated_images.append(batch[0])
        if len(generated_images) >= target_count:
            break

    generated_images = np.array(generated_images)
    generated_labels = np.full(len(generated_images), cls)

    images = np.concatenate([images, generated_images])
    labels = np.concatenate([labels, generated_labels])
    print_class_distribution(labels, "Class distribution after augmentation:")

    return images, to_categorical(labels, num_classes=len(FEATURES))


def split_data(images, labels):
    train_idx, holdout_idx = train_test_split(np.arange(labels.shape[0]), test_size=0.2)
    val_idx, test_idx = train_test_split(holdout_idx, test_size=0.5)

    return (
        images[train_idx], labels[train_idx],
        images[val_idx], labels[val_idx],
        images[test_idx], labels[test_idx],
    )


def preprocess_images(images, size=IMG_SIZE):
    resized = np.array([cv2.resize(img, (size, size)) for img in images])
    return resized.astype(np.float32) / 255.0


def build_model(input_shape, num_classes=len(FEATURES)):
    model = Sequential([
        Conv2D(16, (3, 3), activation="tanh", padding="same", input_shape=input_shape),
        Conv2D(32, (3, 3), activation="tanh", padding="same"),
        MaxPool2D((2, 2)),
        Conv2D(32, (3, 3), activation="tanh", padding="same"),
        MaxPool2D((2, 2)),
        Conv2D(32, (3, 3), activation="tanh", padding="same"),
        MaxPool2D((2, 2)),
        Conv2D(32, (3, 3), activation="tanh", padding="same"),
        MaxPool2D((2, 2)),
        Dropout(0.25),
        Flatten(),
        Dense(32, activation="tanh"),
        Dropout(0.5),
        Dense(num_classes, activation="softmax"),
    ])
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["acc"])
    model.summary()
    return model


def main():
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    images, labels = load_data()
    images, labels = oversample_minority_class(images, labels)

    train_images, train_labels, val_images, val_labels, test_images, test_labels = split_data(images, labels)

    X_train = preprocess_images(train_images)
    X_val = preprocess_images(val_images)
    X_test = preprocess_images(test_images)

    model = build_model(input_shape=X_train.shape[1:])

    early_stopping = EarlyStopping(monitor="val_loss", mode="min", verbose=1, patience=PATIENCE)
    history = model.fit(
        X_train, train_labels,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(X_val, val_labels),
        callbacks=[early_stopping],
    )

    with open(MODEL_JSON_PATH, "w") as json_file:
        json_file.write(model.to_json())
    model.save_weights(MODEL_WEIGHTS_PATH)

    val_loss, val_acc = model.evaluate(X_val, val_labels)
    test_loss, test_acc = model.evaluate(X_test, test_labels)
    print(f"Val Loss: {val_loss:.4f} - Val Accuracy: {val_acc:.4f}")
    print(f"Test Loss: {test_loss:.4f} - Test Accuracy: {test_acc:.4f}")

    plot_accuracy_and_loss_combined(
        history.history["acc"], history.history["val_acc"],
        history.history["loss"], history.history["val_loss"],
        "customCurve.png",
    )

    class_names = list(FEATURES.values())

    val_pred = np.argmax(model.predict(X_val), axis=1)
    val_true = np.argmax(val_labels, axis=1)
    plot_confusion(val_true, val_pred, "custom_val_confusion_matrix.png",
                   title="Validation Confusion Matrix", labels=class_names, cmap="Blues")
    print(classification_report(val_true, val_pred))

    test_pred = np.argmax(model.predict(X_test), axis=1)
    test_true = np.argmax(test_labels, axis=1)
    plot_confusion(test_true, test_pred, "custom_test_confusion_matrix.png",
                   title="Test Confusion Matrix", labels=class_names, cmap="Blues")
    print(classification_report(test_true, test_pred))


if __name__ == "__main__":
    main()
