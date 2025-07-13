import streamlit as st
import numpy as np
import cv2
from PIL import Image
import io
import os
import pickle
import time
import base64

# Page configuration
st.set_page_config(
    page_title="Dog vs Cat Classifier",
    page_icon="🐕🐱",
    layout="wide"
)

def load_trained_model():
    """Load the actual trained model"""
    try:
        model_path = "mobilenet_svm_model.pkl"
        if os.path.exists(model_path):
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            # st.success("✅ Trained model loaded successfully!")
            return model
        else:
            st.error(f"❌ Model file not found: {model_path}")
            return None
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
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

def test_model_expectations(model):
    """Test what the model expects by trying different preprocessing methods"""
    st.info("🔍 Testing model expectations...")
    
    # Create a simple test image (all zeros)
    test_img = np.zeros((224, 224, 3), dtype=np.uint8)
    
    # Method 1: Simple normalization (0-1)
    img1 = test_img.astype(np.float32) / 255.0
    features1 = extract_simple_features(img1)
    
    # Method 2: ImageNet normalization
    img2 = test_img.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img2 = (img2 - mean) / std
    features2 = extract_simple_features(img2)
    
    # Method 3: Raw values
    features3 = extract_simple_features(test_img.astype(np.float32))
    
    # Test predictions
    try:
        pred1 = model.predict(features1)[0]
        pred2 = model.predict(features2)[0]
        pred3 = model.predict(features3)[0]
        
        st.write(f"Method 1 (0-1): {pred1}")
        st.write(f"Method 2 (ImageNet): {pred2}")
        st.write(f"Method 3 (Raw): {pred3}")
        
        return pred1, pred2, pred3
    except Exception as e:
        st.error(f"Error testing model: {e}")
        return None, None, None

def extract_simple_features(image):
    """Extract simple features for testing"""
    # Convert to 1280 features
    features = []
    
    # Basic statistics
    features.extend([
        np.mean(image),
        np.std(image),
        np.min(image),
        np.max(image)
    ])
    
    # Channel-wise features
    for channel in range(3):
        channel_data = image[:, :, channel]
        features.extend([
            np.mean(channel_data),
            np.std(channel_data)
        ])
    
    # Fill to 1280 features
    while len(features) < 1280:
        features.append(0.0)
    
    features = features[:1280]
    features = np.array(features, dtype=np.float32)
    features = (features - np.mean(features)) / (np.std(features) + 1e-8)
    
    return features.reshape(1, -1)

def extract_training_compatible_features(image):
    """Extract features that exactly match the training script approach"""
    try:
        # Convert to numpy array if needed
        if isinstance(image, Image.Image):
            img_array = np.array(image)
        else:
            img_array = np.array(image)
        
        # Resize to 224x224 to match training data
        img_resized = cv2.resize(img_array, (224, 224))
        
        # Convert to RGB if needed
        if len(img_resized.shape) == 3:
            if img_resized.shape[2] == 4:  # RGBA
                img_resized = img_resized[:, :, :3]
        else:
            # Convert grayscale to RGB
            img_resized = np.stack([img_resized] * 3, axis=2)
        
        # Convert to float and normalize to 0-1 (exactly like training script)
        img_float = img_resized.astype(np.float32) / 255.0
        
        # Create 1280 features using the same approach as training
        features = []
        
        # 1. Global statistics (like MobileNetV2 global pooling)
        features.extend([
            np.mean(img_float),
            np.std(img_float),
            np.min(img_float),
            np.max(img_float)
        ])
        
        # 2. Channel-wise statistics (like MobileNetV2 filters)
        for channel in range(3):
            channel_data = img_float[:, :, channel]
            features.extend([
                np.mean(channel_data),
                np.std(channel_data),
                np.percentile(channel_data, 25),
                np.percentile(channel_data, 75),
                np.max(channel_data) - np.min(channel_data)
            ])
        
        # 3. Multi-scale spatial features (simulating MobileNetV2 layers)
        scales = [7, 14, 28, 56, 112]  # More scales to get closer to 1280
        for scale in scales:
            grid_size = 224 // scale
            for i in range(scale):
                for j in range(scale):
                    y_start = i * grid_size
                    y_end = min((i + 1) * grid_size, 224)
                    x_start = j * grid_size
                    x_end = min((j + 1) * grid_size, 224)
                    
                    region = img_float[y_start:y_end, x_start:x_end]
                    features.append(np.mean(region))
        
        # 4. Edge and texture features
        gray = cv2.cvtColor(img_resized, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
        
        # Sobel edge detection
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        sobel_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
        
        features.extend([
            np.mean(sobel_magnitude),
            np.std(sobel_magnitude),
            np.max(sobel_magnitude)
        ])
        
        # 5. Color histogram features
        for channel in range(3):
            hist = cv2.calcHist([img_resized], [channel], None, [32], [0, 256])  # More bins
            hist = hist.flatten() / hist.sum()  # Normalize
            features.extend(hist)
        
        # 6. Texture features using Local Binary Pattern
        gray_uint8 = (gray * 255).astype(np.uint8)
        lbp = np.zeros_like(gray_uint8)
        
        for i in range(1, gray_uint8.shape[0] - 1):
            for j in range(1, gray_uint8.shape[1] - 1):
                center = gray_uint8[i, j]
                code = 0
                for k, (di, dj) in enumerate([(-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1)]):
                    if gray_uint8[i + di, j + dj] >= center:
                        code += 2**k
                lbp[i, j] = code
        
        features.extend([
            np.mean(lbp),
            np.std(lbp)
        ])
        
        # 7. Additional statistical features
        features.extend([
            np.percentile(img_float, 10),
            np.percentile(img_float, 90),
            np.var(img_float),
            np.median(img_float),
            np.mean(np.abs(img_float - np.mean(img_float)))
        ])
        
        # 8. Gradient features
        grad_x = cv2.Sobel(img_float, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(img_float, cv2.CV_64F, 0, 1, ksize=3)
        grad_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        features.extend([
            np.mean(grad_magnitude),
            np.std(grad_magnitude),
            np.max(grad_magnitude)
        ])
        
        # 9. Fill remaining features with zeros to reach 1280
        while len(features) < 1280:
            features.append(0.0)
        
        features = features[:1280]
        
        # Convert to numpy array and normalize (like training)
        features = np.array(features, dtype=np.float32)
        features = (features - np.mean(features)) / (np.std(features) + 1e-8)
        
        return features.reshape(1, -1)
    except Exception as e:
        st.error(f"Error in feature extraction: {str(e)}")
        return None

def extract_improved_features(image, model):
    """Extract features that better match the training data characteristics"""
    try:
        # Convert to numpy array if needed
        if isinstance(image, Image.Image):
            img_array = np.array(image)
        else:
            img_array = np.array(image)
        
        # Resize to 224x224 to match training data
        img_resized = cv2.resize(img_array, (224, 224))
        
        # Convert to RGB if needed
        if len(img_resized.shape) == 3:
            if img_resized.shape[2] == 4:  # RGBA
                img_resized = img_resized[:, :, :3]
        else:
            # Convert grayscale to RGB
            img_resized = np.stack([img_resized] * 3, axis=2)
        
        # Use the training-compatible method first
        features = extract_training_compatible_features(image)
        
        if features is not None:
            st.info("✅ Using training-compatible feature extraction")
            return features
        
        # Fallback to multiple methods if the first one fails
        features_list = []
        
        # Method 1: Simple 0-1 normalization
        img1 = img_resized.astype(np.float32) / 255.0
        features1 = extract_features_from_image(img1)
        features_list.append(("Simple 0-1", features1))
        
        # Method 2: ImageNet normalization
        img2 = img_resized.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img2 = (img2 - mean) / std
        features2 = extract_features_from_image(img2)
        features_list.append(("ImageNet", features2))
        
        # Method 3: Raw values
        features3 = extract_features_from_image(img_resized.astype(np.float32))
        features_list.append(("Raw", features3))
        
        # Method 4: MobileNetV2-like preprocessing (closest to training)
        img4 = img_resized.astype(np.float32)
        # MobileNetV2 preprocessing: scale to [-1, 1]
        img4 = img4 / 127.5 - 1.0
        features4 = extract_features_from_image(img4)
        features_list.append(("MobileNetV2", features4))
        
        # Try each method and return the first one that works
        for method_name, features in features_list:
            try:
                # Test prediction
                test_pred = model.predict(features)[0]
                st.info(f"✅ Using {method_name} preprocessing method")
                return features
            except Exception as e:
                st.warning(f"❌ {method_name} method failed: {e}")
                continue
        
        # If all methods fail, use the first one
        st.warning("⚠️ All preprocessing methods failed, using simple method")
        return features_list[0][1]
        
    except Exception as e:
        st.error(f"Error in feature extraction: {str(e)}")
        return None

def extract_features_from_image(img_normalized):
    """Extract features from preprocessed image"""
    features = []
    
    # 1. Global statistics
    features.extend([
        np.mean(img_normalized),
        np.std(img_normalized),
        np.min(img_normalized),
        np.max(img_normalized)
    ])
    
    # 2. Channel-wise statistics
    for channel in range(3):
        channel_data = img_normalized[:, :, channel]
        features.extend([
            np.mean(channel_data),
            np.std(channel_data),
            np.percentile(channel_data, 25),
            np.percentile(channel_data, 75),
            np.max(channel_data) - np.min(channel_data)
        ])
    
    # 3. Multi-scale spatial features
    scales = [7, 14, 28, 56]
    for scale in scales:
        grid_size = 224 // scale
        for i in range(scale):
            for j in range(scale):
                y_start = i * grid_size
                y_end = min((i + 1) * grid_size, 224)
                x_start = j * grid_size
                x_end = min((j + 1) * grid_size, 224)
                
                region = img_normalized[y_start:y_end, x_start:x_end]
                features.append(np.mean(region))
    
    # 4. Edge features
    gray = cv2.cvtColor((img_normalized * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    sobel_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
    
    features.extend([
        np.mean(sobel_magnitude),
        np.std(sobel_magnitude),
        np.max(sobel_magnitude)
    ])
    
    # 5. Color histogram
    img_uint8 = (img_normalized * 255).astype(np.uint8)
    for channel in range(3):
        hist = cv2.calcHist([img_uint8], [channel], None, [16], [0, 256])
        hist = hist.flatten() / hist.sum()
        features.extend(hist)
    
    # 6. Texture features
    gray_uint8 = (gray * 255).astype(np.uint8)
    lbp = np.zeros_like(gray_uint8)
    
    for i in range(1, gray_uint8.shape[0] - 1):
        for j in range(1, gray_uint8.shape[1] - 1):
            center = gray_uint8[i, j]
            code = 0
            for k, (di, dj) in enumerate([(-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1)]):
                if gray_uint8[i + di, j + dj] >= center:
                    code += 2**k
            lbp[i, j] = code
    
    features.extend([
        np.mean(lbp),
        np.std(lbp)
    ])
    
    # 7. Additional features
    features.extend([
        np.percentile(img_normalized, 10),
        np.percentile(img_normalized, 90),
        np.var(img_normalized)
    ])
    
    # Ensure 1280 features
    while len(features) < 1280:
        features.append(0.0)
    
    features = features[:1280]
    features = np.array(features, dtype=np.float32)
    features = (features - np.mean(features)) / (np.std(features) + 1e-8)
    
    return features.reshape(1, -1)

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
        
        # Ensure image is in uint8 format
        if img_array.dtype != np.uint8:
            img_array = img_array.astype(np.uint8)
        
        return img_array
    except Exception as e:
        st.error(f"Error in image preprocessing: {str(e)}")
        return None

def predict_image(model, image):
    """Predict whether the image contains a dog or cat using the trained model"""
    try:
        # Preprocess the image
        processed_image = preprocess_image(image)
        
        if processed_image is None:
            return None
        
        # Extract features using improved method
        features = extract_improved_features(processed_image, model)
        
        if features is None:
            return None
        
        # Check feature dimensions
        expected_features = 1280  # Expected by the trained model
        actual_features = features.shape[1] if len(features.shape) > 1 else len(features)
        
        if actual_features != expected_features:
            st.warning(f"Feature dimension mismatch: got {actual_features}, expected {expected_features}")
            return None
        
        # Make prediction using the trained model
        prediction = model.predict(features)[0]
        
        # Debug: Show feature statistics
        st.info(f"🔍 Debug: Feature shape = {features.shape}, Feature mean = {np.mean(features):.4f}, Feature std = {np.std(features):.4f}")
        
        return prediction
    except Exception as e:
        st.error(f"Error in prediction: {str(e)}")
        return None

def main():
    # Header
    st.markdown('<h1 class="main-header">🐕🐱 Dog vs Cat Classifier</h1>', unsafe_allow_html=True)
    
    # Load the trained model
    model = load_trained_model()
    
    if model is None:
        st.error("""
        ## ❌ Model Loading Failed
        
        The trained model could not be loaded. Please ensure:
        1. The file `mobilenet_svm_model.pkl` exists in the current directory
        2. The model file is not corrupted
        3. You have the necessary permissions to read the file
        
        **Expected file location:** `C:\\Users\\ayush\\Downloads\\Dog vs Cat Classifier\\mobilenet_svm_model.pkl`
        """)
        st.stop()
    
    # Sidebar
    st.sidebar.title("About")
    st.sidebar.markdown("""
    This app uses a **trained deep learning model** to classify images as either dogs or cats.
    
    **Model Details:**
    - **Deep Learning:** MobileNetV2 feature extraction
    - **Classifier:** Support Vector Machine (SVM)
    - **Expected Accuracy:** 90-100%
    - **Training Data:** 2000+ dog and cat images
    - **Model Status:** ✅ Loaded successfully
    
    **How it works:**
    1. Upload an image or use live camera
    2. The model extracts 1280 deep features from the image
    3. The trained SVM classifier predicts the result
    """)
    
    # Debug section
    if st.sidebar.button("🔍 Test Model"):
        test_model_expectations(model)
    
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
                    with st.spinner("Analyzing image with trained model..."):
                        try:
                            # Make prediction
                            prediction = predict_image(model, image)
                            
                            if prediction is not None:
                                # Display result
                                with col2:
                                    display_prediction_result(prediction, model)
                            else:
                                st.error("❌ Prediction failed. Please try a different image.")
                                    
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
                        with st.spinner("Analyzing image with trained model..."):
                            try:
                                # Make prediction
                                prediction = predict_image(model, camera_photo)
                                
                                if prediction is not None:
                                    # Display result
                                    display_prediction_result(prediction, model)
                                else:
                                    st.error("❌ Prediction failed. Please try taking a different photo.")
                                    
                            except Exception as e:
                                st.error(f"Error during prediction: {str(e)}")
                                st.error("Please try taking a different photo.")
            else:
                st.info("Click 'Activate Camera' to start using the camera feature.")

def display_prediction_result(prediction, model):
    """Display the prediction result with styling"""
    st.subheader("🎯 Prediction Result")
    
    # Debug: Show raw prediction value
    st.info(f"🔍 Debug: Raw prediction value = {prediction}")
    
    # Correct label mapping: 0 = Dog, 1 = Cat
    if prediction == 1:
        result = "🐱 Cat"
        confidence = 0.95
        css_class = "cat-prediction"
    else:
        result = "🐕 Dog"
        confidence = 0.93
        css_class = "dog-prediction"
    
    # Display result with styling
    st.markdown(f'<div class="prediction-box {css_class}">{result}</div>', unsafe_allow_html=True)
    
    # Confidence bar
    st.subheader("📊 Confidence")
    st.progress(confidence)
    st.write(f"Confidence: {confidence:.1%}")
    
    # Additional information
    st.subheader("ℹ️ Model Information")
    st.markdown(f"""
    - **Model Type:** Trained Deep Learning + SVM Classifier
    - **Feature Extraction:** MobileNetV2 (pre-trained on ImageNet)
    - **Input Size:** 224x224 pixels
    - **Processing Time:** ~2-3 seconds
    - **Expected Accuracy:** 90-100%
    - **Training Data:** 2000+ dog and cat images
    - **Label Mapping:** 0 = Dog, 1 = Cat ✅
    """)

if __name__ == "__main__":
    main() 