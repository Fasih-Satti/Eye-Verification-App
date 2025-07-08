# Eye-Verification-App

A desktop-based Eye Movement Verification System using **PyQt5**, **OpenCV**, and **MediaPipe**.  
Users can **register** and **login** by moving their face **left**, **right**, **up**, and **down**.  
It mimics real-world mobile-style facial verification without requiring any key press, using **facial landmarks** for authentication.

## 🔍 Features

- 👁️ Eye detection using MediaPipe Face Mesh
- 💻 GUI built with PyQt5
- 🔐 Secure face-based login system
- 📝 No button press needed — just natural face movements
- 🧠 Registers & verifies face movements: left, right, up, down
- ✅ Real-time feedback and guided pop-ups for movement

## 📦 Requirements

- Python 3.7+
- OpenCV
- MediaPipe
- PyQt5
- NumPy
- pickle (standard library)

Install dependencies using pip:

```bash
pip install opencv-python mediapipe PyQt5 numpy
