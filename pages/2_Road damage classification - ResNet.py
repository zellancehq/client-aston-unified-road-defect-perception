import os
import logging
from pathlib import Path
from typing import NamedTuple
import io

import cv2
import numpy as np
import streamlit as st

# Deep learning framework
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import resnet50, ResNet50_Weights
from PIL import Image
from io import BytesIO

from sample_utils.download import download_file

st.set_page_config(
    page_title="Road damage classification - ResNet",
    page_icon="📷",
    layout="centered",
    initial_sidebar_state="expanded"
)

HERE = Path(__file__).parent
ROOT = HERE.parent

logger = logging.getLogger(__name__)

# ResNet model configuration
class ImageClassifier:
    def __init__(self, model_path=None, num_classes=5):
        self.model_path = model_path
        self.num_classes = num_classes
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        logger.info(f"Initialized ImageClassifier with device: {self.device}")

    def load_model(self):
        """Initialize and load the model"""
        if self.model is not None:
            return

        logger.info("Initializing ResNet50 with ImageNet weights")
        # Use IMAGENET1K_V1 to match training (pretrained=True default)
        self.model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)

        # Freeze feature extractor layers
        logger.info("Freezing feature extraction layers")
        for name, param in self.model.named_parameters():
            if not name.startswith('fc'):
                param.requires_grad = False

        # Replace the fully connected layer
        logger.info(f"Modifying final layer for {self.num_classes} classes")
        in_features = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, self.num_classes),
            nn.Sigmoid()  # Sigmoid is part of the model architecture (matches training)
        )

        # Load pre-trained weights if available
        if self.model_path and os.path.exists(self.model_path):
            try:
                logger.info(f"Attempting to load model weights from {self.model_path}")
                # Try different loading methods
                try:
                    # First try loading with weights_only
                    state_dict = torch.load(self.model_path, map_location=self.device, weights_only=True)
                except Exception as e1:
                    logger.warning(f"Failed to load with weights_only=True: {e1}")
                    # Try without weights_only
                    state_dict = torch.load(self.model_path, map_location=self.device)

                # Handle different state dict formats
                if isinstance(state_dict, dict):
                    if 'state_dict' in state_dict:
                        state_dict = state_dict['state_dict']
                    elif 'model_state_dict' in state_dict:
                        state_dict = state_dict['model_state_dict']

                # Try loading with and without strict mode
                try:
                    self.model.load_state_dict(state_dict, strict=True)
                except Exception as e2:
                    logger.warning(f"Failed to load with strict=True: {e2}")
                    self.model.load_state_dict(state_dict, strict=False)

                logger.info("Successfully loaded model weights")
            except Exception as e:
                logger.error(f"Failed to load model weights: {str(e)}")
                logger.warning("Continuing with ImageNet weights only")

        self.model.to(self.device)
        self.model.eval()
        logger.info("Model successfully initialized and moved to device")

    def predict(self, image_input, threshold=0.5):
        """Make a prediction on an image"""
        # Ensure model is loaded
        if self.model is None:
            self.load_model()

        # Handle different input types
        if isinstance(image_input, bytes):
            # Convert the byte stream to a PIL image
            image = Image.open(io.BytesIO(image_input))
        elif isinstance(image_input, Image.Image):
            image = image_input
        else:
            raise ValueError("Input must be PIL Image or bytes")

        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')

        # Apply the transformations
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)

        # Make the prediction
        with torch.no_grad():
            outputs = self.model(image_tensor)
            # Model already has sigmoid in architecture, so outputs are probabilities
            probabilities = outputs.cpu().numpy().flatten()
            predictions = (probabilities > threshold).astype(float)

        # Class names in alphabetical order (matches training MultiLabelBinarizer sorted order)
        # Training labels: depression_rutting, isolated_crack, patch, pothole, raveling
        class_names = ['Rutting', 'Crack', 'Patch', 'Pothole', 'Raveling']
        predicted_classes = [class_names[i] for i, pred in enumerate(predictions) if pred == 1]
        
        # Debug information
        logger.info(f"Raw outputs: {outputs.cpu().numpy().flatten()}")
        logger.info(f"Probabilities after sigmoid: {probabilities}")
        logger.info(f"Predictions above {threshold}: {predictions}")
        
        if not predicted_classes:  # If no class is predicted above threshold
            # Get the highest probability class
            max_prob_idx = probabilities.argmax()
            predicted_classes = [class_names[max_prob_idx]]
        
        # Create probability dictionary
        all_probabilities = {class_names[i]: float(probabilities[i]) for i in range(len(class_names))}
        
        return {
            "predicted_classes": predicted_classes,
            "primary_class": predicted_classes[0] if predicted_classes else "Unknown",
            "confidence": float(max(probabilities)) if len(predicted_classes) == 1 else float(np.mean([probabilities[class_names.index(cls)] for cls in predicted_classes])),
            "all_probabilities": all_probabilities,
            "raw_probabilities": probabilities.tolist(),
            "model_loaded": self.model_path is not None and os.path.exists(self.model_path) if self.model_path else False
        }

# Model configuration
MODEL_URL = "https://github.com/zellancehq/client-aston-unified-road-defect-perception/releases/download/v1.0/ResNet50_Road_Defects_Model.pth"
MODEL_LOCAL_PATH = ROOT / "./models/ResNet50_Road_Defects_Model.pth"

# Download the model if it doesn't exist
@st.cache_resource
def download_resnet_model():
    try:
        # Check if model directory exists
        model_dir = MODEL_LOCAL_PATH.parent
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # Download model if it doesn't exist
        if not MODEL_LOCAL_PATH.exists():
            with st.spinner("Downloading ResNet model... This may take a few moments."):
                try:
                    download_file(MODEL_URL, MODEL_LOCAL_PATH, expected_size=None)
                    if MODEL_LOCAL_PATH.exists():
                        st.success("✅ ResNet model downloaded successfully!")
                        return str(MODEL_LOCAL_PATH)
                    else:
                        raise Exception("Download completed but file not found")
                except Exception as download_error:
                    st.error(f"❌ Failed to download model: {download_error}")
                    return None
        else:
            return str(MODEL_LOCAL_PATH)
            
    except Exception as e:
        logger.error(f"Failed to download model: {e}")
        st.warning("⚠️ Could not download ResNet model. Using ImageNet weights only.")
        return None

# Initialize ResNet classifier
@st.cache_resource
def load_resnet_model():
    model_path = download_resnet_model()
    return ImageClassifier(model_path=model_path, num_classes=5)

# Load the model
with st.spinner("Loading ResNet model..."):
    resnet_classifier = load_resnet_model()

# Class names in alphabetical order (matches training MultiLabelBinarizer sorted order)
# Training labels: depression_rutting, isolated_crack, patch, pothole, raveling
ROAD_DAMAGE_CLASSES = [
    "Rutting",
    "Crack",
    "Patch", 
    "Pothole",
    "Raveling"
]

st.title("Road Damage Classification - ResNet")

st.write("""
Classify road damage types using a ResNet deep learning model. This approach uses image classification 
to categorize the entire image into different damage types. The model can predict multiple damage types 
simultaneously using multi-label classification.
""")

st.write("Upload an image to classify the type of road damage present.")

# File upload
uploaded_file = st.file_uploader(
    "Choose an image file", 
    type=['png', 'jpg', 'jpeg'],
    help="Upload an image of a road to classify damage type"
)

# Confidence threshold
confidence_threshold = st.slider(
    "Classification Threshold", 
    min_value=0.0, 
    max_value=1.0, 
    value=0.4, 
    step=0.05,
    help="Minimum probability score to consider a class as detected"
)

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    
    # Display image in single column for better UI
    st.subheader("Original Image")
    try:
        st.image(image, caption="Uploaded Image", use_container_width=True)
    except TypeError:
        # Fallback for older Streamlit versions
        st.image(image, caption="Uploaded Image", use_column_width=True)
    
    # Show image info in a more compact way
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Width", f"{image.size[0]}px")
    with col2:
        st.metric("Height", f"{image.size[1]}px")
    with col3:
        st.metric("Mode", image.mode)
    
    st.divider()
    
    st.subheader("Classification Results")
    
    # Perform classification
    with st.spinner("Classifying image using ResNet..."):
        try:
            # Get prediction from ResNet model
            prediction = resnet_classifier.predict(image, threshold=confidence_threshold)
            
            predicted_classes = prediction["predicted_classes"]
            primary_class = prediction["primary_class"]
            confidence = prediction["confidence"]
            all_probs = prediction["all_probabilities"]
            raw_probs = prediction["raw_probabilities"]
            model_loaded = prediction.get("model_loaded", False)
            
            # Show model status
            if model_loaded:
                st.success("🎯 Using fine-tuned ResNet model")
            else:
                st.warning("⚠️ Using ImageNet weights only (not trained for road damage)")
            
            # Display detected classes
            if predicted_classes:
                detected_names = ", ".join(predicted_classes)
                st.success(f"**Detected Classes:** {detected_names}")
            else:
                st.warning("No damage classes detected above threshold")
                # Show highest probability class as fallback
                max_prob_class = max(all_probs.items(), key=lambda x: x[1])
                st.write(f"**Highest Probability:** {max_prob_class[0]} ({max_prob_class[1]:.2%})")
            
            # Display detailed probabilities
            st.subheader("All Class Probabilities")
            
            # Create a detailed results table
            prob_data = []
            for class_name, prob in all_probs.items():
                status = "✅ Detected" if prob >= confidence_threshold else "❌ Below threshold"
                prob_data.append({
                    "Class": class_name,
                    "Probability": f"{prob:.2%}",
                    "Status": status,
                    "Score": prob
                })
            
            # Sort by probability
            prob_data.sort(key=lambda x: x["Score"], reverse=True)
            
            # Display as table
            st.table([{k: v for k, v in item.items() if k != "Score"} for item in prob_data])
            
            # Display as bar chart
            st.subheader("Probability Distribution")
            chart_data = {item["Class"]: item["Score"] for item in prob_data}
            st.bar_chart(chart_data)
            
            # Device info
            device_info = "GPU (CUDA)" if torch.cuda.is_available() else "CPU"
            st.info(f"**Inference Device:** {device_info}")
            
        except Exception as e:
            st.error(f"Error during classification: {str(e)}")
            logger.error(f"Classification error: {e}")
            st.write("**Possible solutions:**")
            st.write("- Ensure the image is a valid road image")
            st.write("- Try a different image format")
            st.write("- Check if the model loaded correctly")
else:
    st.info("👆 Please upload an image to start classification")
