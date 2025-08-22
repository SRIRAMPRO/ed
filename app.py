import os
import time

import av
import cv2
import numpy as np
import streamlit as st
import tensorflow as tf
from streamlit_webrtc import VideoProcessorBase, webrtc_streamer

# -------------------------
# Page setup + your exact CSS/layout
# -------------------------
st.set_page_config(layout="wide")
st.markdown(
    """
    <style>
    .block-container {
        padding-top: 0.5rem;
        padding-bottom: 0.5rem;
    }
    h1 {
        margin-bottom: 25px !important;
        margin-top: 25px !important;
    }
    [data-testid="column"] {
        padding-right: 10px !important;
        padding-left: 10px !important;
        margin-right: 0 !important;
        margin-left: 0 !important;
        height: 480px !important; /* Fixed height for both columns */
        overflow-y: auto; /* Add scroll if content exceeds height */
    }
    [data-testid="column"]:nth-of-type(1) {
        width: 50% !important;
    }
    [data-testid="column"]:nth-of-type(2) {
        width: 50% !important;
    }
    [data-testid="stVideo"] {
        width: 100% !important;
        height: 100% !important; /* Match the column height */
        object-fit: contain; /* Ensure video fits without distorting */
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown("<h1 style='text-align:center;'>🎥 Real-time Emotion Detection </h1>", unsafe_allow_html=True)

# -------------------------
# Safe model loading (works even if original was saved with optimizer)
# -------------------------
@st.cache_resource(show_spinner=True)
def load_emotion_model():
    # Prefer a clean inference copy if you have it
    if os.path.exists("model_infer.keras"):
        return tf.keras.models.load_model("model_infer.keras", compile=False)

    # Fallback to original; disable compile to avoid optimizer restore
    # If this still throws, create a clean copy once (outside Streamlit):
    #   m = tf.keras.models.load_model("model.keras", compile=False)
    #   m.save("model_infer.keras", include_optimizer=False)
    return tf.keras.models.load_model("model.keras", compile=False)

model = load_emotion_model()

# -------------------------
# Your original labels, input size, and face detector
# -------------------------
emotion_labels = [
    "😲 surprise", "😨 fear", "🤢 disgust",
    "😊 happy", "😢 sad", "😠 angry", "😐 neutral"
]
model_input_size = (100, 100)

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# -------------------------
# Create columns exactly like your local app
# -------------------------
col1, col2 = st.columns([2, 2])
video_area = col1.empty()
bars_placeholder = col2.empty()

# -------------------------
# Your original bars renderer (unchanged)
# -------------------------
def render_bars(predictions: np.ndarray):
    if predictions is None or len(predictions) == 0 or np.all(predictions == 0):
        detected_emotion = "None"
        max_idx = None
    else:
        max_idx = int(np.argmax(predictions))
        detected_emotion = emotion_labels[max_idx]

    html_parts = [
        f"<div style='padding:10px; background:#f0f9f0; border-radius:8px; text-align:center; font-size:20px;color:black'><b>✅ Detected Emotion:</b> <span style='color:green;'>{detected_emotion}</span></div>",
        "<h4 style='margin-top:15px;'>📊 Emotion Confidence</h4>"
    ]

    if predictions is None or len(predictions) == 0:
        preds = np.zeros(len(emotion_labels), dtype=float)
    else:
        preds = predictions

    for i, (label, score) in enumerate(zip(emotion_labels, preds)):
        score_float = float(score)
        bar_color = "#4CAF50" if (max_idx is not None and i == max_idx) else "#2196F3"
        html_parts.append(f"""
<div style="margin-bottom:8px;">
    <b>{label}:</b> {score_float*100:.1f}%
    <div style='background:#ddd; border-radius:3px; width:100%; height:18px;'>
        <div style='width:{score_float*100}%; height:18px; background:{bar_color};
            border-radius:3px; transition: width 0.3s ease;'></div>
    </div>
</div>
""")

    bars_placeholder.markdown("".join(html_parts), unsafe_allow_html=True)

# -------------------------
# WebRTC video processor (browser webcam → exact same preprocessing as your local)
# -------------------------
class EmotionProcessor(VideoProcessorBase):
    def __init__(self):
        self.predictions = np.zeros(len(emotion_labels), dtype=float)

    def recv(self, frame):
        # Frame as BGR for OpenCV
        bgr = frame.to_ndarray(format="bgr24")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

        # Detect faces
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        if len(faces) > 0:
            (x, y, w, h) = faces[0]  # first face, like your local code
            # Crop from RGB (your model used RGB)
            face = rgb[y:y+h, x:x+w]
            # Resize to your model's expected input
            face_resized = cv2.resize(face, model_input_size, interpolation=cv2.INTER_AREA)
            # Normalize 0..1 and add batch dimension (1, H, W, 3) exactly like local
            face_normalized = face_resized.astype("float32") / 255.0
            face_input = np.expand_dims(face_normalized, axis=0)
            # Predict
            preds = model.predict(face_input, verbose=0)[0]
            self.predictions = preds.astype(float)

            # Draw rectangle for UX (optional; doesn't change layout)
            cv2.rectangle(bgr, (x, y), (x+w, y+h), (0, 255, 0), 2)
        else:
            self.predictions = np.zeros(len(emotion_labels), dtype=float)

        # Return the BGR frame to display
        return av.VideoFrame.from_ndarray(bgr, format="bgr24")

# -------------------------
# Start WebRTC in the left column
# -------------------------
with col1:
    ctx = webrtc_streamer(
        key="emotion-detection",
        video_processor_factory=EmotionProcessor,
        media_stream_constraints={"video": True, "audio": False},
        # video_html_attrs={"style": "height:480px; width:auto; object-fit:cover;"}
    )

# -------------------------
# Live-bars updater loop (main thread) – keeps your right column in sync
# -------------------------
if ctx and ctx.video_processor:
    placeholder = st.empty()
    for _ in range(1000):  # arbitrary large number
        if not ctx.state.playing:
            break
        preds = getattr(ctx.video_processor, "predictions", None)
        render_bars(preds)
        time.sleep(0.1)
else:
    # Initial empty state
    render_bars(np.zeros(len(emotion_labels), dtype=float))
