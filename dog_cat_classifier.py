#!/usr/bin/env python3
"""
Dog vs Cat Classifier using MobileNetV2 and SVM
This script downloads the dataset from Kaggle, preprocesses images, extracts features using MobileNetV2,
and trains an SVM classifier for binary classification.
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
import zipfile
import subprocess
import sys

# Parameters
image_size = (224, 224)
dataset_path = "dogs-vs-cats/images/train/train"
MAX_IMAGES = 2000

def setup_kaggle():
    """Setup Kaggle API for dataset download"""
    print("Setting up Kaggle API...")
    
    # Check if kaggle.json exists
    if not os.path.exists("kaggle.json"):
        print("kaggle.json not found. Please upload your kaggle.json file.")
        return False
    
    # Create .kaggle directory
    kaggle_dir = os.path.expanduser("~/.kaggle")
    os.makedirs(kaggle_dir, exist_ok=True)
    
    # Copy kaggle.json to .kaggle directory
    import shutil
    shutil.copy("kaggle.json", os.path.join(kaggle_dir, "kaggle.json"))
    
    # Set permissions (Unix-like systems)
    try:
        os.chmod(os.path.join(kaggle_dir, "kaggle.json"), 0o600)
    except:
        pass  # Windows doesn't support chmod
    
    # Install kaggle if not already installed
    try:
        import kaggle
    except ImportError:
        print("Installing kaggle package...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "kaggle"])
    
    return True

def download_dataset():
    """Download the dogs-vs-cats dataset from Kaggle"""
    print("Downloading dataset from Kaggle...")
    
    try:
        import kaggle
        kaggle.api.authenticate()
        kaggle.api.competition_download_files('dogs-vs-cats', path='.')
        print("Dataset downloaded successfully!")
        
        # Extract the zip files
        if os.path.exists("train.zip"):
            print("Extracting training data...")
            import zipfile
            with zipfile.ZipFile("train.zip", 'r') as zip_ref:
                zip_ref.extractall("dogs-vs-cats/images/train")
            print("Training data extracted successfully!")
            
        if os.path.exists("test1.zip"):
            print("Extracting test data...")
            with zipfile.ZipFile("test1.zip", 'r') as zip_ref:
                zip_ref.extractall("dogs-vs-cats/images/test")
            print("Test data extracted successfully!")
            
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        print("Please make sure you have a valid kaggle.json file and internet connection.")
        return False
    
    return True

def load_images(folder, max_images=None):
    """Load and preprocess images from the specified folder"""
    images = []
    labels = []
    
    if not os.path.exists(folder):
        print(f"Dataset folder not found: {folder}")
        return None, None
    
    filenames = [f for f in os.listdir(folder) if f.lower().endswith('.jpg')]
    filenames = shuffle(filenames, random_state=42)

    if max_images:
        filenames = filenames[:max_images]

    print(f"Loading {len(filenames)} images...")
    
    for i, fname in enumerate(filenames):
        if i % 100 == 0:
            print(f"   Progress: {i}/{len(filenames)}")
            
        label = 0 if "cat" in fname.lower() else 1
        path = os.path.join(folder, fname)
        img = cv2.imread(path)
        if img is not None:
            img = cv2.resize(img, image_size)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = img_to_array(img)
            images.append(img)
            labels.append(label)
    
    print(f"Loaded {len(images)} images successfully.")
    return np.array(images), np.array(labels)

def extract_features(images):
    """Extract deep features using MobileNetV2"""
    print("Loading MobileNetV2 model...")
    model = MobileNetV2(weights='imagenet', include_top=False, pooling='avg', input_shape=(224, 224, 3))
    
    print("Extracting features...")
    images = preprocess_input(images)
    features = model.predict(images, batch_size=32, verbose=1)
    
    print(f"Extracted features shape: {features.shape}")
    return features

def plot_samples(images, labels, preds, label_map, n=10):
    """Plot sample predictions"""
    plt.figure(figsize=(20, 4))
    for i in range(min(n, len(images))):
        ax = plt.subplot(1, n, i + 1)
        plt.imshow(images[i].astype("uint8"))
        plt.title(f"True: {label_map[labels[i]]}\nPred: {label_map[preds[i]]}")
        plt.axis("off")
    plt.tight_layout()
    plt.show()

def save_model(model, filename="mobilenet_svm_model.pkl"):
    """Save the trained model"""
    print(f"Saving model to {filename}...")
    with open(filename, 'wb') as f:
        pickle.dump(model, f)
    print("Model saved successfully!")

def load_model(filename="mobilenet_svm_model.pkl"):
    """Load a trained model"""
    if os.path.exists(filename):
        print(f"Loading model from {filename}...")
        with open(filename, 'rb') as f:
            model = pickle.load(f)
        print("Model loaded successfully!")
        return model
    else:
        print(f"Model file not found: {filename}")
        return None

def main():
    """Main function to run the complete pipeline"""
    print("Dog vs Cat Classifier")
    print("=" * 50)
    
    # Check if dataset exists
    if not os.path.exists(dataset_path):
        print("Dataset not found. Setting up Kaggle and downloading...")
        if not setup_kaggle():
            print("Failed to setup Kaggle. Exiting.")
            return
        
        if not download_dataset():
            print("Failed to download dataset. Exiting.")
            return
    
    # Load and preprocess images
    print("Loading and preprocessing images...")
    images, labels = load_images(dataset_path, max_images=MAX_IMAGES)
    
    if images is None:
        print("Failed to load images. Exiting.")
        return
    
    images, labels = shuffle(images, labels, random_state=42)

    # Extract features
    print("Extracting deep features using MobileNetV2...")
    if len(images) == 0:
        print("No images found. Please check if the dataset was extracted correctly.")
        return
    features = extract_features(images)

    # Split dataset
    print("Splitting dataset...")
    X_train, X_test, y_train, y_test, img_train, img_test = train_test_split(
        features, labels, images, test_size=0.2, random_state=42
    )

    # Train SVM
    print("Training SVM classifier...")
    svm = SVC(kernel='rbf', C=1.0, gamma='scale')
    svm.fit(X_train, y_train)

    # Predict
    print("Making predictions...")
    y_pred = svm.predict(X_test)

    # Evaluate
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {acc * 100:.2f}%")

    # Confusion matrix
    print("Generating confusion matrix...")
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Cat", "Dog"])
    disp.plot()
    plt.title("Confusion Matrix - Dog vs Cat Classification")
    plt.show()

    # Visualize predictions
    print("Visualizing sample predictions...")
    label_map = {0: 'Cat', 1: 'Dog'}
    plot_samples(img_test[:10], y_test[:10], y_pred[:10], label_map)

    # Save model
    save_model(svm)

    print("Training completed successfully!")

def predict_single_image(model, image_path):
    """Predict a single image using the trained model"""
    if model is None:
        print("No model loaded.")
        return
    
    # Load and preprocess image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Could not load image: {image_path}")
        return
    
    img = cv2.resize(img, image_size)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    
    # Extract features
    features = extract_features(img)
    
    # Predict
    prediction = model.predict(features)[0]
    label_map = {0: 'Cat', 1: 'Dog'}
    
    print(f"Prediction: {label_map[prediction]}")
    return prediction

if __name__ == "__main__":
    # Check if model already exists
    if os.path.exists("mobilenet_svm_model.pkl"):
        print("Found existing model. Loading...")
        model = load_model()
        if model is not None:
            print("Model loaded successfully!")
            print("You can use predict_single_image(model, 'path/to/image.jpg') to predict new images.")
        else:
            print("Training new model...")
            main()
    else:
        print("No existing model found. Training new model...")
        main() 