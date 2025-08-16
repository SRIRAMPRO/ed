import cv2
import numpy as np
import streamlit as st
import tensorflow as tf

st.set_page_config(layout="wide")
st.markdown(
    """
    <style>
    /* Reduce overall container padding */
    .block-container {
        padding-top: 0.5rem;
        padding-bottom: 0.5rem;
    }
    /* Add gap below the title */
    h1 {
        margin-bottom: 25px !important; /* adjust gap between title and containers */
        margin-top: 25px !important; 
    }
    /* Remove column padding completely */
    [data-testid="column"] {
        padding-right: 0 !important;
        padding-left: 0 !important;
        margin-right: 0 !important;
        margin-left: 0 !important;
    }
    /* Pull first column closer to second */
    [data-testid="column"]:nth-of-type(1) {
        margin-right: -100px !important; /* shrink gap between camera and detection */
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown("<h1 style='text-align:center;'>🎥 Real-time Emotion Detection 😃</h1>", unsafe_allow_html=True)

# Load model
model = tf.keras.models.load_model("model.keras")

# Emotion labels with emojis
emotion_labels = [
    "😲 surprise", "😨 fear", "🤢 disgust",
    "😊 happy", "😢 sad", "😠 angry", "😐 neutral"
]
model_input_size = (100, 100)

# Load face cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Create columns with closer spacing
col1, col2 = st.columns([2.8, 1])  # slightly closer ratio

video_placeholder = col1.empty()
bars_placeholder = col2.empty()

# Open webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    st.error("🚫 Could not open webcam")
    st.stop()

# A helper function to render bars with highlight on max
# A helper function to render bars with highlight on max
def render_bars(predictions):
    if np.all(predictions == 0):
        detected_emotion = "None"
        max_idx = None
    else:
        max_idx = np.argmax(predictions)
        detected_emotion = emotion_labels[max_idx]

    html_parts = [
        f"<div style='padding:10px; background:#f0f9f0; border-radius:8px; text-align:center; font-size:20px;color:black'><b>✅ Detected Emotion:</b> <span style='color:green;'>{detected_emotion}</span></div>",
        "<h4 style='margin-top:15px;'>📊 Emotion Confidence</h4>"
    ]

    for i, (label, score) in enumerate(zip(emotion_labels, predictions)):
        score_float = float(score)
        bar_color = "#4CAF50" if max_idx is not None and i == max_idx else "#2196F3"
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

while True:
    ret, frame = cap.read()
    if not ret:
        st.warning("⚠️ Failed to grab frame")
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
    if len(faces) > 0:
        (x, y, w, h) = faces[0]
        face = rgb_frame[y:y+h, x:x+w]
        face_resized = cv2.resize(face, model_input_size)
        face_normalized = face_resized.astype("float32") / 255.0
        face_input = np.expand_dims(face_normalized, axis=0)
        predictions = model.predict(face_input, verbose=0)[0]
    else:
        predictions = np.zeros(len(emotion_labels))

    video_placeholder.image(rgb_frame, channels="RGB")
    render_bars(predictions)

    # Streamlit refresh hack: Stop infinite loop if Streamlit reruns app
    if not st.session_state.get('run', True):
        break