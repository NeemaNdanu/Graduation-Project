import streamlit as st
from PIL import Image
import base64
from io import BytesIO
import os

from video_detect import run_object_detection
from speed_estimation import run_speed_estimation

# --- App Config ---
st.set_page_config(
    page_title="Traffic Intelligence App",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Base64 Image Embedding ---
def get_base64_image(image_path):
    img = Image.open(image_path)
    buffered = BytesIO()
    img.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode()

# --- Custom CSS Styling ---
st.markdown("""
    <style>
        body, .stApp {
            background-color: #eaf6ff;
        }

        .main-title {
            text-align: center;
            font-size: 32px;
            color: #003366;
            font-weight: bold;
            margin-top: 10px;
        }

        .tagline {
            text-align: center;
            font-size: 20px;
            color: #cc0000;
            margin-bottom: 30px;
        }

        .center-img img {
            display: block;
            margin-left: auto;
            margin-right: auto;
            width: 100%;
            height: 300px;
            object-fit: cover;
            border-radius: 15px;
            box-shadow: 2px 2px 10px rgba(0, 0, 0, 0.2);
        }

        .stButton>button {
            background-color: #003366;
            color: white;
            border-radius: 8px;
            font-weight: bold;
        }

        .stButton>button:hover {
            background-color: #b30000;
        }

        .download-button {
            text-align: center;
            margin-top: 20px;
        }

        .block {
            background-color: #ffffff;
            padding: 20px;
            border-radius: 10px;
            border: 1px solid #ccc;
            margin-bottom: 20px;
        }
    </style>
""", unsafe_allow_html=True)

# --- Top Banner Image ---
img_path = "C:/Users/HP/Downloads/street at night.jpeg"
img_base64 = get_base64_image(img_path)
st.markdown(f"""
    <div class="center-img">
        <img src="data:image/jpeg;base64,{img_base64}">
    </div>
""", unsafe_allow_html=True)

# --- Title and Tagline ---
st.markdown('<div class="main-title">🚗 Smart Road Eye – Real-Time Traffic Analyzer</div>', unsafe_allow_html=True)
st.markdown('<div class="tagline">Empowering Smarter Roads with Real-Time Vision</div>', unsafe_allow_html=True)

# --- Sidebar: Mode Selection ---
with st.sidebar:
    mode = st.selectbox(
        "Select Analysis Mode:",
        ["Object Detection", "Speed Estimation"],
        index=0
    )

# --- Video Upload and Processing ---
st.markdown('<div class="block">', unsafe_allow_html=True)
video_file = st.file_uploader("🎥 Upload Road Traffic Video (MP4 only)", type=["mp4"])

if video_file:
    input_path = "videos/temp_video.mp4"
    output_path = "videos/output_video.mp4"

    # Save uploaded file
    with open(input_path, "wb") as f:
        f.write(video_file.read())

    st.success("✅ Video uploaded successfully! Processing...")

    model_path = "C:/Users/HP/Desktop/Data science Project/best2.pt"

    # --- Run the selected mode (using only positional arguments) ---
    if mode == "Object Detection":
        run_object_detection(input_path, output_path, model_path)
    elif mode == "Speed Estimation":
        run_speed_estimation(input_path, output_path, model_path)
    
    # --- Show processed video ---
    st.video(output_path)

    # --- Download button ---
    with open(output_path, "rb") as f:
        video_bytes = f.read()

    st.markdown('<div class="download-button">', unsafe_allow_html=True)
    st.download_button(
        label="⬇️ Download Processed Video",
        data=video_bytes,
        file_name="processed_traffic_video.mp4",
        mime="video/mp4"
    )
    st.markdown('</div>', unsafe_allow_html=True)

else:
    st.info("📌 Please upload a video to begin analysis.")

st.markdown('</div>', unsafe_allow_html=True) 

