import sys
import cv2
import os
import pickle
import numpy as np
import mediapipe as mp
from PyQt5.QtWidgets import (
    QApplication, QWidget, QPushButton, QLabel, QVBoxLayout, QLineEdit, QMessageBox
)
from PyQt5.QtCore import Qt

LEFT_EYE_IDX = [33, 160, 158, 133, 153]
RIGHT_EYE_IDX = [362, 385, 387, 263, 373]
DB_FILE = "eye_user_data.pkl"

if not os.path.exists(DB_FILE):
    with open(DB_FILE, "wb") as f:
        pickle.dump({}, f)

def save_user(name, eye_crop):
    with open(DB_FILE, "rb") as f:
        db = pickle.load(f)
    db[name.lower()] = eye_crop
    with open(DB_FILE, "wb") as f:
        pickle.dump(db, f)

def get_user(name):
    with open(DB_FILE, "rb") as f:
        db = pickle.load(f)
    return db.get(name.lower())

def extract_eyes(image, face_mesh):
    result = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    if not result.multi_face_landmarks:
        return None, image, None, None
    h, w, _ = image.shape
    landmarks = result.multi_face_landmarks[0]
    eyes = []
    nose_x = int(landmarks.landmark[1].x * w)
    nose_y = int(landmarks.landmark[1].y * h)
    for eye_idx in [LEFT_EYE_IDX, RIGHT_EYE_IDX]:
        pts = [(int(landmarks.landmark[i].x * w), int(landmarks.landmark[i].y * h)) for i in eye_idx]
        x1, y1 = min(p[0] for p in pts), min(p[1] for p in pts)
        x2, y2 = max(p[0] for p in pts), max(p[1] for p in pts)
        crop = image[y1:y2, x1:x2]
        if crop.size == 0:
            return None, image, None, None
        crop = cv2.resize(crop, (50, 50))
        eyes.append(crop)
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    if len(eyes) == 2:
        return np.hstack(eyes), image, nose_x, nose_y
    return None, image, None, None

class EyeVerificationApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Eye Verification App")
        self.setGeometry(200, 200, 300, 250)
        layout = QVBoxLayout()

        self.label = QLabel("Eye Verification System", self)
        self.label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.label)

        self.name_input = QLineEdit(self)
        self.name_input.setPlaceholderText("Enter your name")
        layout.addWidget(self.name_input)

        self.register_btn = QPushButton("Register", self)
        self.register_btn.clicked.connect(self.register_user)
        layout.addWidget(self.register_btn)

        self.login_btn = QPushButton("Login", self)
        self.login_btn.clicked.connect(self.login_user)
        layout.addWidget(self.login_btn)

        self.setLayout(layout)

    def register_user(self):
        name = self.name_input.text().strip()
        if not name:
            QMessageBox.warning(self, "Input Error", "Please enter a name.")
            return

        if get_user(name) is not None:
            QMessageBox.warning(self, "Already Exists", "User already registered.")
            return

        cap = cv2.VideoCapture(0)
        moved_left = moved_right = moved_up = moved_down = False
        stored_eyes = None
        center_x = center_y = None

        with mp.solutions.face_mesh.FaceMesh(refine_landmarks=True) as face_mesh:

            QMessageBox.information(self, "Step 1", "Please move your face to the LEFT")
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frame = cv2.flip(frame, 1)
                eyes, output, nose_x, nose_y = extract_eyes(frame, face_mesh)
                cv2.imshow("Register - Move Left", output)

                if eyes is not None and nose_x is not None:
                    if center_x is None:
                        center_x = nose_x
                    if nose_x - center_x < -30:
                        moved_left = True
                        stored_eyes = eyes
                        break
                if cv2.waitKey(1) & 0xFF == 27:
                    break

            if moved_left:
                QMessageBox.information(self, "Step 2", "Now move your face to the RIGHT")
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    frame = cv2.flip(frame, 1)
                    eyes, output, nose_x, nose_y = extract_eyes(frame, face_mesh)
                    cv2.imshow("Register - Move Right", output)

                    if eyes is not None and nose_x is not None:
                        if nose_x - center_x > 30:
                            moved_right = True
                            break
                    if cv2.waitKey(1) & 0xFF == 27:
                        break

            if moved_left and moved_right:
                QMessageBox.information(self, "Step 3", "Now move your face UP")
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    frame = cv2.flip(frame, 1)
                    eyes, output, nose_x, nose_y = extract_eyes(frame, face_mesh)
                    cv2.imshow("Register - Move Up", output)

                    if eyes is not None and nose_y is not None:
                        if center_y is None:
                            center_y = nose_y
                        if nose_y - center_y < -20:
                            moved_up = True
                            break
                    if cv2.waitKey(1) & 0xFF == 27:
                        break

            if moved_up:
                QMessageBox.information(self, "Step 4", "Now move your face DOWN")
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    frame = cv2.flip(frame, 1)
                    eyes, output, nose_x, nose_y = extract_eyes(frame, face_mesh)
                    cv2.imshow("Register - Move Down", output)

                    if eyes is not None and nose_y is not None:
                        if nose_y - center_y > 20:
                            moved_down = True
                            break
                    if cv2.waitKey(1) & 0xFF == 27:
                        break

        cap.release()
        cv2.destroyAllWindows()

        if all([moved_left, moved_right, moved_up, moved_down]) and stored_eyes is not None:
            eye_crop = cv2.resize(stored_eyes, (100, 50))
            save_user(name, eye_crop)
            QMessageBox.information(self, "Success", "✅ Registration completed successfully.")
        else:
            QMessageBox.warning(self, "Failed", "❌ Movement not detected correctly.")

    def login_user(self):
        name = self.name_input.text().strip()
        if not name:
            QMessageBox.warning(self, "Input Error", "Please enter a name.")
            return

        stored = get_user(name)
        if stored is None:
            QMessageBox.critical(self, "Error", "User not found.")
            return

        cap = cv2.VideoCapture(0)
        center_x = None
        moved_left = moved_right = False

        with mp.solutions.face_mesh.FaceMesh(refine_landmarks=True) as face_mesh:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frame = cv2.flip(frame, 1)
                eyes, output, nose_x, _ = extract_eyes(frame, face_mesh)
                cv2.imshow("Login", output)

                if eyes is not None and nose_x is not None:
                    eye_crop = cv2.resize(eyes, (100, 50))
                    diff = np.mean(np.abs(eye_crop.astype("float") - stored.astype("float")))
                    if diff < 25:
                        if center_x is None:
                            center_x = nose_x
                        offset = nose_x - center_x
                        if offset < -30:
                            moved_left = True
                        elif offset > 30 and moved_left:
                            moved_right = True
                            break
                if cv2.waitKey(1) & 0xFF == 27:
                    break

        cap.release()
        cv2.destroyAllWindows()

        if moved_left and moved_right:
            QMessageBox.information(self, "Success", f"✅ Welcome back, {name}! Verification completed.")
        else:
            QMessageBox.critical(self, "Failed", "❌ Face movement not completed properly.")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = EyeVerificationApp()
    window.show()
    sys.exit(app.exec_())
