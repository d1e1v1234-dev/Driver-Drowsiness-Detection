import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import av

st.set_page_config(page_title="Driver Drowsiness Detection", layout="centered")
st.title("🚗 Driver Drowsiness Detection")
st.write("Real-time eye and yawn monitoring using CNN models (MobileNetV3).")

# =========================================================
# CONFIGURATION
# =========================================================
IMG_SIZE = (128, 128)
EYE_CLOSED_THRESHOLD = 3
YAWN_THRESHOLD = 0.5

EYE_MODEL_PATH = "models/eye_model_fixed.keras"
YAWN_MODEL_PATH = "models/yawn_model_fixed.keras"

eye_class_names = ["Closed_Eyes", "Open_Eyes"]
yawn_class_names = ["No_yawn", "Yawn"]

# =========================================================
# LOAD MODELS (cached so it only loads once)
# =========================================================
@st.cache_resource
def load_models():
    model_eye = tf.keras.models.load_model(EYE_MODEL_PATH, compile=False)
    model_yawn = tf.keras.models.load_model(YAWN_MODEL_PATH, compile=False)
    return model_eye, model_yawn

model_eye, model_yawn = load_models()

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")

if face_cascade.empty() or eye_cascade.empty():
    st.error("Failed to load Haar cascades.")
    st.stop()

# =========================================================
# PREPROCESS IMAGE (matches your local script exactly)
# =========================================================
def preprocess_image(image):
    image = cv2.resize(image, IMG_SIZE, interpolation=cv2.INTER_AREA)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.astype(np.float32)
    image = np.expand_dims(image, axis=0)
    return image

RTC_CONFIGURATION = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})

# =========================================================
# VIDEO PROCESSOR
# =========================================================
class DrowsinessProcessor(VideoProcessorBase):
    def __init__(self):
        self.closed_eye_counter = 0

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80)
        )

        status_text = "No face detected"
        status_color = (200, 200, 200)

        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 0), 2)

            face_gray = gray[y:y + h, x:x + w]
            face_color = img[y:y + h, x:x + w]

            eye_label = "Unknown"
            yawn_label = "Unknown"

            # --- Eye detection ---
            eyes = eye_cascade.detectMultiScale(
                face_gray, scaleFactor=1.1, minNeighbors=5, minSize=(20, 20)
            )

            if len(eyes) > 0:
                ex, ey, ew, eh = eyes[0]
                eye_crop = face_color[ey:ey + eh, ex:ex + ew]

                if eye_crop.size > 0:
                    eye_arr = preprocess_image(eye_crop)
                    eye_pred = model_eye.predict(eye_arr, verbose=0)[0][0]
                    eye_index = int(eye_pred > 0.5)
                    eye_label = eye_class_names[eye_index]

                    cv2.rectangle(img, (x + ex, y + ey), (x + ex + ew, y + ey + eh), (0, 255, 0), 2)

                    if eye_label == "Closed_Eyes":
                        self.closed_eye_counter += 1
                    else:
                        self.closed_eye_counter = 0

            # --- Yawn detection ---
            mouth_y_start = int(h * 0.60)
            mouth_crop = face_color[mouth_y_start:h, 0:w]

            if mouth_crop.size > 0:
                mouth_arr = preprocess_image(mouth_crop)
                yawn_pred = model_yawn.predict(mouth_arr, verbose=0)[0][0]
                yawn_index = int(yawn_pred > YAWN_THRESHOLD)
                yawn_label = yawn_class_names[yawn_index]

                cv2.rectangle(img, (x, y + mouth_y_start), (x + w, y + h), (255, 0, 255), 2)

            # --- Drowsiness decision ---
            if self.closed_eye_counter >= EYE_CLOSED_THRESHOLD or yawn_label == "Yawn":
                status_text = "DROWSY ALERT!"
                status_color = (0, 0, 255)
            else:
                status_text = f"Alert | Eyes: {eye_label} | Mouth: {yawn_label}"
                status_color = (0, 255, 0)

            break  # only process first face

        cv2.putText(img, status_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
        cv2.putText(img, f"Closed Eye Frames: {self.closed_eye_counter}", (20, 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# =========================================================
# STREAMLIT UI
# =========================================================
webrtc_streamer(
    key="drowsiness-detection",
    video_processor_factory=DrowsinessProcessor,
    rtc_configuration=RTC_CONFIGURATION,
    media_stream_constraints={"video": True, "audio": False},
)

st.markdown("---")
st.markdown("**Note:** Grant camera access when prompted. Detection runs entirely in-browser via WebRTC.")