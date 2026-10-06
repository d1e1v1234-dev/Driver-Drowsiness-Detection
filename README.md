# 😴 Driver Drowsiness Detection

Ever nodded off for a second while driving? This app catches that moment before it turns into an accident.

It watches your face through the webcam, and if your eyes stay closed for too long, it yells at you with an alarm. Built with Python, OpenCV and Streamlit.

👉 **Try it live:** [add your Streamlit link here]

---

## 🔥 What it does

- Watches your face in real time (webcam or video)
- Checks if your eyes are open or closed [edit: EAR / CNN / yawn detection]
- Plays an alarm + shows a warning when you look sleepy
- Runs in the browser with a clean Streamlit UI, no complicated setup

## 🧰 Built with

- **Python**
- **OpenCV** for video + face handling
- **[MediaPipe / dlib / TensorFlow – edit this]** for eye detection
- **Streamlit** for the web app

## 🧠 How it works (quick version)

1. Grab a frame from the camera
2. Find the face and eyes
3. Measure how open the eyes are
4. Eyes closed for too many frames in a row? You're drowsy 🚨
5. Alarm goes off

## 🚀 Run it yourself

```bash
# 1. Clone it
git clone https://github.com/d1e1v1234-dev/Driver-Drowsiness-Detection.git
cd Driver-Drowsiness-Detection

# 2. Install stuff
pip install -r requirements.txt

# 3. Start the app
streamlit run app.py
```

Open `http://localhost:8501` and you're good to go.

## 📁 Files

```
├── app.py             # the Streamlit app
├── requirements.txt   # dependencies
├── [model / alarm files]
└── README.md
```

## 📸 Screenshots

[add 1-2 screenshots or a GIF here, makes the repo look 10x better]

## ⚠️ Heads up

- Works best with good lighting and a clear view of your face
- Sunglasses can confuse it
- It's a helper, not a replacement for taking a break when you're tired. Please pull over and rest 🙏

## 🚧 Next up

- Yawn detection
- Head-tilt detection
- Make it work on mobile

## 👋 Made by

**Dev**, 3rd year CSE student at IIIT Una.
Found a bug or have an idea? Open an issue or send a PR!

⭐ If this helped you, drop a star!
