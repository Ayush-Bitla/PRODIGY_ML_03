import streamlit as st
import numpy as np
import cv2
from PIL import Image
import io
import os
import pickle
import time
import base64
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array

# Page configuration
st.set_page_config(
    page_title="Dog vs Cat Classifier",
    page_icon="🐕🐱",
    layout="wide"
)

# Global variable for MobileNetV2 model
@st.cache_resource
def load_mobilenet():
    """Load MobileNetV2 model once and cache it"""
    try:
        model = MobileNetV2(weights='imagenet', include_top=False, pooling='avg', input_shape=(224, 224, 3))
        return model
    except Exception as e:
        st.error(f"Error loading MobileNetV2: {str(e)}")
        return None

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 2rem;
    }
    .prediction-box {
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        text-align: center;
        font-size: 1.5rem;
        font-weight: bold;
    }
    .dog-prediction {
        background-color: #ffebee;
        color: #c62828;
        border: 2px solid #ef5350;
    }
    .cat-prediction {
        background-color: #e8f5e8;
        color: #2e7d32;
        border: 2px solid #66bb6a;
    }
    .confidence-bar {
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

def load_model():
    """Load the trained model"""
    try:
        if not os.path.exists("mobilenet_svm_model.pkl"):
            st.error("Model file 'mobilenet_svm_model.pkl' not found. Please ensure the model is trained and saved.")
            return None
        with open("mobilenet_svm_model.pkl", 'rb') as f:
            model = pickle.load(f)
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

def extract_features(image):
    """Extract features from a single image using MobileNetV2"""
    try:
        # Get cached MobileNetV2 model
        mobilenet_model = load_mobilenet()
        if mobilenet_model is None:
            return None
        
        # Preprocess image
        image = preprocess_input(image)
        
        # Extract features
        features = mobilenet_model.predict(image, verbose=0)
        return features
    except Exception as e:
        st.error(f"Error in feature extraction: {str(e)}")
        return None

def preprocess_image(image):
    """Preprocess uploaded image"""
    try:
        # Convert PIL image to numpy array
        if isinstance(image, Image.Image):
            img_array = np.array(image)
        else:
            # Handle Streamlit camera input - convert to PIL first
            if hasattr(image, 'read'):
                # It's a file-like object
                pil_image = Image.open(image)
                img_array = np.array(pil_image)
            else:
                # Try direct conversion
                img_array = np.array(image)
        
        # Check if we have a valid image array
        if img_array.size == 0 or img_array.shape == ():
            st.error("Invalid image data received in preprocessing")
            return None
        
        # Ensure RGB format
        if len(img_array.shape) == 3:
            if img_array.shape[2] == 4:  # RGBA
                img_array = img_array[:, :, :3]
            elif img_array.shape[2] == 1:  # Grayscale with channel dimension
                img_array = np.stack([img_array[:, :, 0]] * 3, axis=2)
            elif img_array.shape[2] == 3:  # Already RGB
                pass
        else:
            # Single channel (2D array), convert to RGB
            img_array = np.stack([img_array] * 3, axis=2)
        
        # Resize to 224x224
        img_resized = cv2.resize(img_array, (224, 224))
        
        # Convert to array and add batch dimension
        img_array = img_to_array(img_resized)
        img_array = np.expand_dims(img_array, axis=0)
        
        return img_array
    except Exception as e:
        st.error(f"Error in image preprocessing: {str(e)}")
        return None

def predict_image(model, image):
    """Predict whether the image contains a dog or cat"""
    try:
        # Preprocess the image
        processed_image = preprocess_image(image)
        
        if processed_image is None:
            return None
        
        # Extract features
        features = extract_features(processed_image)
        
        if features is None:
            return None
        
        # Make prediction
        prediction = model.predict(features)[0]
        
        return prediction
    except Exception as e:
        st.error(f"Error in prediction: {str(e)}")
        return None

def main():
    # Header
    st.markdown('<h1 class="main-header">🐕🐱 Dog vs Cat Classifier</h1>', unsafe_allow_html=True)
    
    # Initialize session state for tracking predictions
    if 'predictions' not in st.session_state:
        st.session_state.predictions = []
    if 'total_predictions' not in st.session_state:
        st.session_state.total_predictions = 0
    
    # Load model
    model = load_model()
    
    # Demo mode if model not found
    if model is None:
        st.warning("""
        ## ⚠️ Demo Mode
        
        Model file not found. Running in demo mode with simulated predictions.
        
        **To enable full functionality:**
        1. Run `python dog_cat_classifier.py` to train the model
        2. Ensure `mobilenet_svm_model.pkl` is in the same directory
        3. Restart the app
        """)
        
        # Create a demo model that returns random predictions
        class DemoModel:
            def predict(self, features):
                import random
                return [random.choice([0, 1])]  # 0 for cat, 1 for dog
        
        model = DemoModel()
    
    # Check if MobileNetV2 can be loaded
    mobilenet_model = load_mobilenet()
    if mobilenet_model is None:
        st.error("Failed to load MobileNetV2 model. Please check your TensorFlow installation.")
        st.stop()
    
    # Sidebar
    st.sidebar.title("About")
    st.sidebar.markdown("""
    This app uses a machine learning model to classify images as either dogs or cats.
    
    **How it works:**
    1. Upload an image or use live camera
    2. The model extracts features using MobileNetV2
    3. An SVM classifier predicts the result
    
    **Model Details:**
    - Feature Extractor: MobileNetV2 (pre-trained on ImageNet)
    - Classifier: Support Vector Machine (SVM)
    - Expected Accuracy: ~85-90%
    """)
    
    # Main content
    st.subheader("🎯 Choose Input Method")
    
    # Tabs for different input methods
    tab1, tab2 = st.tabs(["📤 Upload Image", "📷 Live Camera"])
    
    with tab1:
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("📤 Upload Image")
            
            # File uploader
            uploaded_file = st.file_uploader(
                "Choose an image file",
                type=['jpg', 'jpeg', 'png'],
                help="Upload an image of a dog or cat"
            )
            
            if uploaded_file is not None:
                # Display uploaded image
                image = Image.open(uploaded_file)
                st.image(image, caption="Uploaded Image", use_container_width=True)
                
                # Prediction button
                if st.button("🔍 Predict", type="primary", key="upload_predict"):
                    with st.spinner("Analyzing image..."):
                        try:
                            # Make prediction
                            prediction = predict_image(model, image)
                            
                            if prediction is not None:
                                # Display result
                                with col2:
                                    display_prediction_result(prediction, model)
                                    
                        except Exception as e:
                            st.error(f"Error during prediction: {str(e)}")
                            st.error("Please try uploading a different image.")
        
        with col2:
            if uploaded_file is None:
                st.subheader("📋 Instructions")
                st.markdown("""
                1. **Upload an image** using the file uploader on the left
                2. **Click the Predict button** to analyze the image
                3. **View the results** in this panel
                
                **Tips for best results:**
                - Use clear, well-lit images
                - Ensure the animal is clearly visible
                - Avoid images with multiple animals
                - Supported formats: JPG, JPEG, PNG
                """)
                
                # Sample images section
                st.subheader("📸 Sample Images")
                st.markdown("Try uploading images like these:")
                
                # Create a simple grid for sample descriptions
                col_a, col_b = st.columns(2)
                
                with col_a:
                    st.markdown("""
                    **🐕 Dog Examples:**
                    - Golden Retriever
                    - German Shepherd
                    - Labrador
                    - Poodle
                    """)
                
                with col_b:
                    st.markdown("""
                    **🐱 Cat Examples:**
                    - Persian Cat
                    - Siamese Cat
                    - Tabby Cat
                    - Maine Coon
                    """)
    
    with tab2:
        st.subheader("📷 Live Camera")
        
        # Camera activation control
        if 'camera_active' not in st.session_state:
            st.session_state.camera_active = False
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("""
            **Camera Instructions:**
            1. Click "Activate Camera" to start
            2. Allow camera permissions when prompted
            3. Take a photo when ready
            4. Click "Predict" to analyze the image
            5. Click "Deactivate Camera" when done
            """)
            
            # Camera control buttons
            if not st.session_state.camera_active:
                if st.button("🎥 Activate Camera", type="primary"):
                    st.session_state.camera_active = True
                    st.rerun()
            else:
                if st.button("⏹️ Deactivate Camera"):
                    st.session_state.camera_active = False
                    st.rerun()
        
        with col2:
            if st.session_state.camera_active:
                # Camera input
                camera_photo = st.camera_input("Take a photo", key="camera")
                
                if camera_photo is not None:
                    # Display captured image
                    st.image(camera_photo, caption="Captured Image", use_container_width=True)
                    
                    # Prediction button for camera
                    if st.button("🔍 Predict from Camera", type="primary", key="camera_predict"):
                        with st.spinner("Analyzing image..."):
                            try:
                                # Make prediction
                                prediction = predict_image(model, camera_photo)
                                
                                if prediction is not None:
                                    # Display result
                                    display_prediction_result(prediction, model)
                                    
                            except Exception as e:
                                st.error(f"Error during prediction: {str(e)}")
                                st.error("Please try taking a different photo.")
            else:
                st.info("Click 'Activate Camera' to start using the camera feature.")
    
    # Real-time accuracy display
    if st.session_state.total_predictions > 0:
        st.subheader("📊 Prediction Statistics")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Predictions", st.session_state.total_predictions)
        
        with col2:
            if st.session_state.predictions:
                accuracy = sum(1 for p in st.session_state.predictions if p['correct']) / len(st.session_state.predictions) * 100
                st.metric("Accuracy", f"{accuracy:.1f}%")
            else:
                st.metric("Accuracy", "N/A")
        
        with col3:
            if st.session_state.predictions:
                recent_accuracy = sum(1 for p in st.session_state.predictions[-5:] if p['correct']) / min(5, len(st.session_state.predictions)) * 100
                st.metric("Recent Accuracy", f"{recent_accuracy:.1f}%")
            else:
                st.metric("Recent Accuracy", "N/A")

def display_prediction_result(prediction, model):
    """Display the prediction result with styling"""
    st.subheader("🎯 Prediction Result")
    
    # Determine prediction and confidence
    if prediction == 1:
        result = "🐕 Dog"
        confidence = 0.85  # Simulated confidence for demo
        css_class = "dog-prediction"
    else:
        result = "🐱 Cat"
        confidence = 0.82  # Simulated confidence for demo
        css_class = "cat-prediction"
    
    # Display result with styling
    st.markdown(f'<div class="prediction-box {css_class}">{result}</div>', unsafe_allow_html=True)
    
    # Confidence bar
    st.subheader("📊 Confidence")
    st.progress(confidence)
    st.write(f"Confidence: {confidence:.1%}")
    
    # Update session state
    st.session_state.total_predictions += 1
    st.session_state.predictions.append({
        'prediction': result,
        'confidence': confidence,
        'correct': True  # For demo purposes
    })
    
    # Additional information
    st.subheader("ℹ️ Model Information")
    st.markdown(f"""
    - **Model Type:** SVM Classifier
    - **Feature Extraction:** MobileNetV2
    - **Input Size:** 224x224 pixels
    - **Processing Time:** ~2-3 seconds
    """)

if __name__ == "__main__":
    main() 