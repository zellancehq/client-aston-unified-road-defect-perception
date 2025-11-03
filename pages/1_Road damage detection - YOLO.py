import os
import logging
from pathlib import Path
from typing import NamedTuple

import cv2
import numpy as np
import streamlit as st

# Deep learning framework
from ultralytics import YOLO
from PIL import Image
from io import BytesIO

from sample_utils.download import download_file

st.set_page_config(
    page_title="Road damage detection - YOLOv12",
    page_icon="📷",
    layout="centered",
    initial_sidebar_state="expanded"
)

HERE = Path(__file__).parent
ROOT = HERE.parent

logger = logging.getLogger(__name__)

MODEL_URL = "https://github.com/achilis1505/RoadDamageDetection/raw/main/models/YOLOv12_Road_Defects_Model.pt"  # noqa: E501
MODEL_LOCAL_PATH = ROOT / "./models/YOLOv12_Road_Defects_Model.pt"

# Download the model if it doesn't exist
@st.cache_resource
def download_yolo_model():
    try:
        # Check if model directory exists
        model_dir = MODEL_LOCAL_PATH.parent
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # Download model if it doesn't exist
        if not MODEL_LOCAL_PATH.exists():
            with st.spinner("Downloading YOLOv12 model... This may take a few moments."):
                try:
                    download_file(MODEL_URL, MODEL_LOCAL_PATH, expected_size=None)
                    if MODEL_LOCAL_PATH.exists():
                        st.success("✅ YOLOv12 model downloaded successfully!")
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
        st.error("⚠️ Could not download YOLOv12 model.")
        return None

# Load the YOLO model
@st.cache_resource
def load_yolo_model():
    try:
        model_path = download_yolo_model()
        if model_path:
            return YOLO(model_path)
        return None
    except Exception as e:
        logger.error(f"Failed to load YOLO model: {e}")
        st.error(f"⚠️ Error loading YOLO model: {str(e)}")
        return None

# Load the model
with st.spinner("Loading YOLOv12 model..."):
    net = load_yolo_model()

CLASSES = [
    "alligator cracking",  # Index 0
    "linear cracking",     # Index 1
    "patching",            # Index 2
    "pothole",             # Index 3
    "rutting"              # Index 4
]

class Detection(NamedTuple):
    class_id: int
    label: str
    score: float
    box: np.ndarray

st.title("Road Damage Detection - YOLOv12")

st.write("""
Detect road damage using a YOLOv12 deep learning model. This approach uses object detection 
to identify and locate specific damage types in road images with bounding boxes around detected areas.
""")

st.write("Upload an image to detect road damage and get precise locations of damage areas.")

# File upload
image_file = st.file_uploader(
    "Choose an image file", 
    type=['png', 'jpg', 'jpeg'],
    help="Upload an image of a road to detect damage"
)

# Confidence threshold
score_threshold = st.slider(
    "Confidence Threshold", 
    min_value=0.0, 
    max_value=1.0, 
    value=0.15, 
    step=0.05,
    help="Lower the threshold if no damage is detected, increase if there are false predictions"
)

if image_file is not None and net is not None:
    try:
        # Load the image
        image = Image.open(image_file)
        
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
        
        st.subheader("Detection Results")
        
        # Perform detection
        with st.spinner("Detecting road damage using YOLOv12..."):
            try:
                # Perform inference
                # Convert image to RGB if it's not already (handles grayscale images)
                if image.mode != 'RGB':
                    image = image.convert('RGB')
                
                _image = np.array(image)
                h_ori = _image.shape[0]
                w_ori = _image.shape[1]

                image_resized = cv2.resize(_image, (640, 640), interpolation=cv2.INTER_AREA)
                results = net.predict(image_resized, conf=score_threshold)
                
                # Save the results
                detections = []
                for result in results:
                    boxes = result.boxes.cpu().numpy()
                    detections = [
                       Detection(
                           class_id=int(_box.cls),
                           label=CLASSES[int(_box.cls)],
                           score=float(_box.conf),
                           box=_box.xyxy[0].astype(int),
                        )
                        for _box in boxes
                    ]

                annotated_frame = results[0].plot()
                _image_pred = cv2.resize(annotated_frame, (w_ori, h_ori), interpolation=cv2.INTER_AREA)
                
                # Show detection summary
                if detections:
                    st.success(f"🎯 **{len(detections)} damage(s) detected**")
                    
                    # Group detections by class
                    damage_counts = {}
                    for detection in detections:
                        if detection.label in damage_counts:
                            damage_counts[detection.label] += 1
                        else:
                            damage_counts[detection.label] = 1
                    
                    # Display detected damage types in a green box
                    damage_list = "\n\n".join([f"• ***{damage_type}: {count} instance(s)***" for damage_type, count in damage_counts.items()])
                    st.success(f"**Detected Damage Types:**\n\n{damage_list}")
                    
                    # Show detailed detections
                    with st.expander("📋 Detailed Detection Results"):
                        for i, detection in enumerate(detections, 1):
                            st.write(f"**Detection {i}:**")
                            st.write(f"- Type: {detection.label}")
                            st.write(f"- Confidence: {detection.score:.2%}")
                            st.write(f"- Bounding Box: {detection.box}")
                            st.divider()
                else:
                    st.warning("⚠️ No damage detected above the confidence threshold")
                    st.info("💡 Try lowering the confidence threshold if you expect damage to be present")
                
                # Display the prediction image
                st.subheader("Annotated Image")
                try:
                    st.image(_image_pred, caption="Image with detected damage highlighted", use_container_width=True)
                except TypeError:
                    # Fallback for older Streamlit versions
                    st.image(_image_pred, caption="Image with detected damage highlighted", use_column_width=True)

                # Download predicted image
                buffer = BytesIO()
                _downloadImages = Image.fromarray(_image_pred)
                _downloadImages.save(buffer, format="PNG")
                _downloadImagesByte = buffer.getvalue()

                st.download_button(
                    label="📥 Download Prediction Image",
                    data=_downloadImagesByte,
                    file_name="RDD_Prediction.png",
                    mime="image/png",
                    help="Download the image with detected damage annotations"
                )
                
                # Device info
                device_info = "GPU (CUDA)" if hasattr(net, 'device') and 'cuda' in str(net.device) else "CPU"
                st.info(f"**Inference Device:** {device_info}")
                
            except Exception as e:
                st.error(f"Error during detection: {str(e)}")
                logger.error(f"Detection error: {e}")
                st.write("**Possible solutions:**")
                st.write("- Ensure the image is a valid road image")
                st.write("- Try a different image format")
                st.write("- Check if the model loaded correctly")
    
    except Exception as e:
        st.error(f"Error processing uploaded image: {str(e)}")
        st.write("Please try uploading a different image file.")

elif image_file is not None and net is None:
    st.error("❌ YOLOv12 model failed to load. Please check your internet connection and try refreshing the page.")
elif image_file is None:
    st.info("👆 Please upload an image to start detection")