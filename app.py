import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import av
import time

st.set_page_config(page_title="Driver Drowsiness Detection", layout="centered")
st.title("🚗 Driver Drowsiness Detection")
st.write("Real-time eye and yawn monitoring using CNN models (MobileNetV3).")

# =========================================================
# CONFIGURATION
# =========================================================
IMG_SIZE = (128, 128)
EYE_CLOSED_THRESHOLD = 5
YAWN_THRESHOLD = 0.5

EYE_MODEL_PATH = "models/eye_model_fixed.keras"
YAWN_MODEL_PATH = "models/yawn_model_fixed.keras"

eye_class_names = ["Closed_Eyes", "Open_Eyes"]
yawn_class_names = ["No_yawn", "Yawn"]

# =========================================================
# LOAD MODELS
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
# PREPROCESS IMAGE
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
        self.frame_count = 0
        self.predict_every_n = 3
        self.last_eye_label = "Unknown"
        self.last_yawn_label = "Unknown"
        self.drowsy_start_time = None
        self.is_drowsy_long = False

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        self.frame_count += 1
        run_prediction = (self.frame_count % self.predict_every_n == 0)

        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80))
        status_text = "No face detected"
        status_color = (200, 200, 200)

        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 0), 2)
            face_gray = gray[y:y+h, x:x+w]
            face_color = img[y:y+h, x:x+w]

            if run_prediction:
                eyes = eye_cascade.detectMultiScale(face_gray, minSize=(20, 20))
                if len(eyes) > 0:
                    ex, ey, ew, eh = eyes[0]
                    eye_crop = face_color[ey:ey+eh, ex:ex+ew]
                    if eye_crop.size > 0:
                        eye_arr = preprocess_image(eye_crop)
                        eye_pred = model_eye.predict(eye_arr, verbose=0)[0][0]
                        self.last_eye_label = eye_class_names[int(eye_pred > 0.5)]
                        cv2.rectangle(img, (x+ex, y+ey), (x+ex+ew, y+ey+eh), (0,255,0), 2)
                        if self.last_eye_label == "Closed_Eyes":
                            self.closed_eye_counter += 1
                        else:
                            self.closed_eye_counter = 0

                mouth_y_start = int(h * 0.6)
                mouth_crop = face_color[mouth_y_start:h, 0:w]
                if mouth_crop.size > 0:
                    mouth_arr = preprocess_image(mouth_crop)
                    yawn_pred = model_yawn.predict(mouth_arr, verbose=0)[0][0]
                    self.last_yawn_label = yawn_class_names[int(yawn_pred > YAWN_THRESHOLD)]
                    cv2.rectangle(img, (x, y+mouth_y_start), (x+w, y+h), (255,0,255), 2)

            eye_label = self.last_eye_label
            yawn_label = self.last_yawn_label

            is_drowsy_now = self.closed_eye_counter >= EYE_CLOSED_THRESHOLD or yawn_label == "Yawn"

            if is_drowsy_now:
                if self.drowsy_start_time is None:
                    self.drowsy_start_time = time.time()
                elif time.time() - self.drowsy_start_time >= 5:
                    self.is_drowsy_long = True
                status_text = "DROWSY ALERT!"
                status_color = (0, 0, 255)
            else:
                self.drowsy_start_time = None
                self.is_drowsy_long = False
                status_text = f"Alert | Eyes: {eye_label} | Mouth: {yawn_label}"
                status_color = (0, 255, 0)

            break

        cv2.putText(img, status_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
        cv2.putText(img, f"Closed Eye Frames: {self.closed_eye_counter}", (20, 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# =========================================================
# STREAMLIT UI
# =========================================================
st.markdown("---")
st.markdown("**Note:** Grant camera access when prompted. Detection runs entirely in-browser via WebRTC.")

webrtc_ctx = webrtc_streamer(
    key="drowsiness-detection",
    video_processor_factory=DrowsinessProcessor,
    rtc_configuration=RTC_CONFIGURATION,
    media_stream_constraints={
        "video": {"width": {"ideal": 480}, "height": {"ideal": 360}},
        "audio": False
    },
)

# =========================================================
# SOUND ALERT (plays once per drowsy episode ≥5s)
# =========================================================
alert_placeholder = st.empty()

if webrtc_ctx.state.playing:
    while True:
        if webrtc_ctx.video_processor:

            if webrtc_ctx.video_processor.is_drowsy_long:

                # Keep alarm playing continuously
                alert_placeholder.markdown(
                    """
                    <audio autoplay loop>
                        <source src="https://actions.google.com/sounds/v1/alarms/beep_short.ogg"
                                type="audio/ogg">
                    </audio>
                    """,
                    unsafe_allow_html=True
                )
            else:
                alert_placeholder.empty()

        time.sleep(0.1)