# Driver Drowsiness Detection

Real-time drowsiness detection using two CNN classifiers (eye-state and yawn-detection, MobileNetV3 transfer learning) combined with OpenCV face/eye detection and Streamlit WebRTC for live video.

## Run locally
\`\`\`bash
pip install -r requirements.txt
streamlit run app.py
\`\`\`

## Deploy
Hosted on Hugging Face Spaces using Docker SDK.