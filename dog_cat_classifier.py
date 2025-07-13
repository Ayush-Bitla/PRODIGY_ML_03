#!/usr/bin/env python3
"""
Dog vs Cat Classifier using MobileNetV2 + SVM
"""

import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.utils import shuffle
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
import pickle

# Parameters
image_size = (224, 224)
dataset_path = "dogs-vs-cats/images/train/train"
MAX_IMAGES = 2000
MODEL_FILENAME = "mobilenet_svm_model.pkl"

def load_images(folder, max_images=None):
    images, labels = [], []

    if not os.path.exists(folder):
        print(f"Dataset folder not found: {folder}")
        return None, None

    filenames = [f for f in os.listdir(folder) if f.lower().endswith('.jpg')]
    filenames = shuffle(filenames, random_state=42)
    if max_images:
        filenames = filenames[:max_images]

    for i, fname in enumerate(filenames):
        if i % 100 == 0:
            print(f"Loading images: {i}/{len(filenames)}")
        label = 0 if "cat" in fname.lower() else 1
        path = os.path.join(folder, fname)
        img = cv2.imread(path)
        if img is not None:
            img = cv2.resize(img, image_size)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = img_to_array(img)
            images.append(img)
            labels.append(label)

    return np.array(images), np.array(labels)

def extract_features(images):
    model = MobileNetV2(weights='imagenet', include_top=False, pooling='avg', input_shape=(224, 224, 3))
    images = preprocess_input(images)
    features = model.predict(images, batch_size=32, verbose=1)
    return features

def save_model(model, filename=MODEL_FILENAME):
    with open(filename, 'wb') as f:
        pickle.dump(model, f)

def load_model(filename=MODEL_FILENAME):
    if os.path.exists(filename):
        with open(filename, 'rb') as f:
            return pickle.load(f)
    else:
        return None

def plot_predictions(images, labels, preds, label_map, filename="sample_predictions.png"):
    plt.figure(figsize=(20, 4))
    for i in range(min(10, len(images))):
        ax = plt.subplot(1, 10, i + 1)
        plt.imshow(images[i].astype("uint8"))
        plt.title(f"T: {label_map[labels[i]]}\nP: {label_map[preds[i]]}", fontsize=10)
        plt.axis("off")
    plt.tight_layout()
    plt.savefig(filename)

def train_model():
    images, labels = load_images(dataset_path, max_images=MAX_IMAGES)
    if images is None or len(images) == 0:
        print("No images loaded. Check dataset path.")
        return

    images, labels = shuffle(images, labels, random_state=42)
    features = extract_features(images)

    X_train, X_test, y_train, y_test, img_train, img_test = train_test_split(
        features, labels, images, test_size=0.2, random_state=42)

    svm = SVC(kernel='rbf', C=1.0, gamma='scale')
    svm.fit(X_train, y_train)

    y_pred = svm.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Test Accuracy: {acc * 100:.2f}%")

    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Cat", "Dog"])
    disp.plot()
    plt.title("Confusion Matrix")
    plt.savefig("confusion_matrix.png")

    label_map = {0: 'Cat', 1: 'Dog'}
    plot_predictions(img_test[:10], y_test[:10], y_pred[:10], label_map)

    save_model(svm)
    print("Training complete. Model and plots saved.")

def predict_single_image(image_path, model=None):
    if model is None:
        model = load_model()
        if model is None:
            print("Model not found.")
            return

    img = cv2.imread(image_path)
    if img is None:
        print(f"Could not read image: {image_path}")
        return

    img = cv2.resize(img, image_size)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)

    mobilenet = MobileNetV2(weights='imagenet', include_top=False, pooling='avg', input_shape=(224, 224, 3))
    features = mobilenet.predict(img)
    prediction = model.predict(features)[0]

    label_map = {0: 'Cat', 1: 'Dog'}
    print(f"Predicted label: {label_map[prediction]}")
    return prediction

if __name__ == "__main__":
    if os.path.exists(MODEL_FILENAME):
        print("Model already exists. Use predict_single_image(path) to classify new images.")
    else:
        train_model()
