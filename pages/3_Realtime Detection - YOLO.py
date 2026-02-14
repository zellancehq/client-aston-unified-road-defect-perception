import logging
import queue
from pathlib import Path
from typing import List, NamedTuple

import av
import cv2
import numpy as np
import streamlit as st
from streamlit_webrtc import WebRtcMode, webrtc_streamer

# Deep learning framework
from ultralytics import YOLO

from sample_utils.download import download_file
from sample_utils.get_STUNServer import getSTUNServer

st.set_page_config(
    page_title="Realtime Road Damage Detection - YOLOv12",
    page_icon="📹",
    layout="centered",
    initial_sidebar_state="expanded"
)

HERE = Path(__file__).parent
ROOT = HERE.parent

logger = logging.getLogger(__name__)

MODEL_URL = "https://github.com/zellancehq/client-aston-unified-road-defect-perception/releases/download/v1.0/YOLOv12_Road_Defects_Model.pt"  # noqa: E501
MODEL_LOCAL_PATH = ROOT / "./models/YOLOv12_Road_Defects_Model.pt"

# STUN/TURN Server configuration with reliable fallbacks
def get_ice_servers():
    """Get ICE servers with reliable public STUN servers as fallback"""
    ice_servers = [
        {"urls": ["stun:stun.l.google.com:19302"]},
        {"urls": ["stun:stun1.l.google.com:19302"]},
        {"urls": ["stun:stun2.l.google.com:19302"]},
        {"urls": ["stun:stun3.l.google.com:19302"]},
        {"urls": ["stun:stun4.l.google.com:19302"]},
    ]
    
    # Try to add closest STUN server
    try:
        closest_stun = getSTUNServer()
        if closest_stun:
            ice_servers.insert(0, {"urls": [f"stun:{closest_stun}"]})
    except Exception as e:
        logger.warning(f"Could not get closest STUN server: {e}")
    
    return ice_servers

ICE_SERVERS = get_ice_servers()

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

st.title("Realtime Road Damage Detection - YOLOv12")

st.write("""
Detect road damage in realtime using your webcam. This feature uses YOLOv12 deep learning model 
for on-site monitoring with personnel on the ground. Select the video input device and start detection.
""")

if net is None:
    st.error("❌ YOLOv12 model failed to load. Please check your internet connection and try refreshing the page.")
    st.stop()

# Confidence threshold - must be defined BEFORE the callback that uses it
score_threshold = st.slider(
    "Confidence Threshold", 
    min_value=0.0, 
    max_value=1.0, 
    value=0.15, 
    step=0.05,
    help="Lower the threshold if no damage is detected, increase if there are false predictions"
)

st.info("💡 Lower the threshold if there is no damage detected, and increase if there are false predictions.")

# NOTE: The callback will be called in another thread,
#       so use a queue here for thread-safety to pass the data
#       from inside to outside the callback.
# TODO: A general-purpose shared state object may be more useful.
result_queue: "queue.Queue[List[Detection]]" = queue.Queue()

def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
    
    image = frame.to_ndarray(format="bgr24")
    h_ori = image.shape[0]
    w_ori = image.shape[1]
    image_resized = cv2.resize(image, (640, 640), interpolation = cv2.INTER_AREA)
    results = net.predict(image_resized, conf=score_threshold)
    
    # Save the results on the queue
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
        result_queue.put(detections)

    annotated_frame = results[0].plot()
    _image = cv2.resize(annotated_frame, (w_ori, h_ori), interpolation = cv2.INTER_AREA)

    return av.VideoFrame.from_ndarray(_image, format="bgr24")

webrtc_ctx = webrtc_streamer(
    key="road-damage-detection",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration={"iceServers": ICE_SERVERS},
    video_frame_callback=video_frame_callback,
    media_stream_constraints={
        "video": {
            "width": {"ideal": 1280, "min": 800},
        },
        "audio": False
    },
    async_processing=True,
)

st.divider()

if st.checkbox("Show Predictions Table", value=False):
    if webrtc_ctx.state.playing:
        labels_placeholder = st.empty()
        while True:
            result = result_queue.get()
            labels_placeholder.table(result)