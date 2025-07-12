# 🐕🐱 Dog vs Cat Classifier (PRODIGY_ML_03)

A comprehensive machine learning project for binary classification of dog and cat images using MobileNetV2 for feature extraction and Support Vector Machine (SVM) for classification. Built for the Prodigy ML internship.

## 🎯 Project Overview

This project implements a state-of-the-art image classification system that can distinguish between dogs and cats with high accuracy. The system uses transfer learning with MobileNetV2 pre-trained on ImageNet for feature extraction, followed by an SVM classifier for final prediction.

## 🚀 Live Demo

Access the live Streamlit app here:  
👉 [Dog vs Cat Classifier App](https://dog-cat-classifier.streamlit.app/)

## 🧠 Features

### Core ML Features
- **Transfer Learning**: Uses MobileNetV2 pre-trained on ImageNet
- **Feature Extraction**: Deep features from MobileNetV2 (1280 dimensions)
- **Classification**: Support Vector Machine (SVM) with RBF kernel
- **High Accuracy**: ~85-90% accuracy on test set
- **Real-time Processing**: Fast inference for live predictions

### Web Application Features
- **📤 Upload Images**: Drag & drop or click to upload images
- **📷 Live Camera**: Real-time camera capture and prediction
- **🎯 Instant Predictions**: Get results in 2-3 seconds
- **📊 Real-time Accuracy**: Dynamic confidence scoring
- **💾 Model Download**: Save trained models for later use
- **🔄 Auto-prediction**: Optional automatic prediction on camera capture

### Technical Features
- **Dataset Management**: Automatic Kaggle dataset download
- **Model Persistence**: Save/load trained models
- **Error Handling**: Robust error handling for invalid images
- **Cross-platform**: Works on Windows, Mac, and Linux
- **Virtual Environment**: Isolated dependencies

## 📦 Installation & Setup

### Prerequisites
- Python 3.7+
- Internet connection for dataset download
- Kaggle API credentials

### Quick Start

1. **Clone the repository:**
```bash
git clone https://github.com/Ayush-Bitla/PRODIGY_ML_03.git
cd PRODIGY_ML_03
```

2. **Create virtual environment:**
```bash
python -m venv .venv
.venv\Scripts\Activate.ps1  # Windows
# source .venv/bin/activate  # Mac/Linux
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

4. **Setup Kaggle API:**
   - Go to [Kaggle](https://www.kaggle.com/) and create an account
   - Go to your account settings and create an API token
   - Download the `kaggle.json` file
   - Place the `kaggle.json` file in the project directory

5. **Run the main script:**
```bash
python dog_cat_classifier.py
```

6. **Launch the web app:**
```bash
streamlit run app.py
```

## 🎮 How to Use

### Training the Model
1. Run `python dog_cat_classifier.py`
2. The script will automatically:
   - Download the Dogs vs Cats dataset from Kaggle
   - Load and preprocess 2000 images
   - Extract features using MobileNetV2
   - Train an SVM classifier
   - Show accuracy and confusion matrix
   - Save the trained model

### Using the Web App
1. **Upload Tab**: Upload images from your device
2. **Camera Tab**: Use live camera for real-time predictions
3. **Activate Camera**: Click to start camera (privacy-friendly)
4. **Auto-predict**: Enable for instant results
5. **View Results**: See predictions with confidence scores

### Making Predictions
```python
from dog_cat_classifier import load_model, predict_single_image

# Load the trained model
model = load_model()

# Predict a single image
prediction = predict_single_image(model, "path/to/image.jpg")
# Returns: 0 for Cat, 1 for Dog
```

## 📊 Model Performance

### Training Results
- **Dataset**: Dogs vs Cats (2000 images)
- **Feature Extractor**: MobileNetV2 (ImageNet weights)
- **Classifier**: SVM with RBF kernel
- **Accuracy**: ~85-90%
- **Training Time**: ~5-10 minutes
- **Inference Time**: ~2-3 seconds per image

### Model Architecture
```
Input Image (224x224x3)
    ↓
MobileNetV2 (Feature Extraction)
    ↓
Global Average Pooling
    ↓
1280-dimensional Features
    ↓
SVM Classifier (RBF kernel)
    ↓
Binary Prediction (Cat/Dog)
```

## 🛠️ Project Structure

```
PRODIGY_ML_03/
├── dog_cat_classifier.py    # Main training script
├── app.py                   # Streamlit web application
├── requirements.txt         # Python dependencies
├── README.md               # This file
├── kaggle.json            # Kaggle API credentials (you add this)
├── mobilenet_svm_model.pkl # Trained model (generated after training)
├── dogs-vs-cats/          # Dataset folder (downloaded automatically)
│   └── images/
│       ├── train/         # Training images
│       └── test/          # Test images
└── .venv/                 # Virtual environment
```

## 📈 What We Built

### 1. **Complete ML Pipeline**
- ✅ Dataset download and preprocessing
- ✅ Feature extraction using MobileNetV2
- ✅ SVM training and evaluation
- ✅ Model saving and loading
- ✅ Prediction on new images

### 2. **Interactive Web Application**
- ✅ Modern UI with tabs and real-time features
- ✅ File upload functionality
- ✅ Live camera integration
- ✅ Real-time accuracy tracking
- ✅ Error handling and user feedback

### 3. **Advanced Features**
- ✅ Transfer learning with pre-trained models
- ✅ Batch processing for efficiency
- ✅ Cross-platform compatibility
- ✅ Virtual environment setup
- ✅ Comprehensive documentation

### 4. **Production-Ready Features**
- ✅ Model persistence
- ✅ Error handling
- ✅ Progress tracking
- ✅ User-friendly interface
- ✅ Scalable architecture

## 🔧 Technical Details

### Dependencies
- **TensorFlow**: Deep learning framework
- **OpenCV**: Image processing
- **Scikit-learn**: Machine learning algorithms
- **Streamlit**: Web application framework
- **Pandas**: Data manipulation
- **Matplotlib**: Visualization
- **Kaggle**: Dataset download

### Model Specifications
- **Input Size**: 224x224x3 RGB images
- **Feature Dimensions**: 1280 (MobileNetV2 output)
- **Classifier**: SVM with RBF kernel
- **Optimization**: Grid search for hyperparameters
- **Regularization**: Built into SVM

## 🎯 Key Learnings

### Machine Learning
- Transfer learning with pre-trained models
- Feature extraction vs end-to-end training
- Hyperparameter tuning for SVM
- Model evaluation and validation
- Data preprocessing and augmentation

### Software Engineering
- Modular code design
- Error handling and logging
- User interface development
- Model deployment and serving
- Version control and documentation

### Web Development
- Streamlit framework usage
- Real-time data processing
- Camera integration
- State management
- Responsive design

## 🤝 Contributing

Feel free to submit issues and enhancement requests!

## 📝 License

This project is part of the PRODIGY ML internship program.

## 🙏 Acknowledgments

- **Dataset**: [Dogs vs Cats](https://www.kaggle.com/c/dogs-vs-cats) competition on Kaggle
- **Model**: MobileNetV2 from TensorFlow/Keras
- **Framework**: Streamlit for web application
- **Mentorship**: PRODIGY ML internship program

---

**Built with ❤️ for the PRODIGY ML internship** 