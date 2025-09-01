# ml_model.py
import cv2
import numpy as np

class MotionClassifier:
    def __init__(self, model_path=None):
        """
        Initialize ML model.
        For now, we’ll just use a simple placeholder.
        Later you can load a deep learning model (YOLO, MobileNet, etc.)
        """
        self.model = None
        if model_path:
            # Example for future: self.model = cv2.dnn.readNet(model_path)
            print(f"[INFO] Loaded model from {model_path}")
        else:
            print("[INFO] Using placeholder motion classifier")

    def classify_motion(self, frame, contour):
        """
        Classify type of motion (currently placeholder).
        contour: the detected moving object’s contour.
        frame: the video frame.
        """
        x, y, w, h = cv2.boundingRect(contour)
        roi = frame[y:y+h, x:x+w]

        # Placeholder: if width > height, say 'horizontal motion'
        if w > h:
            return "Horizontal Motion"
        else:
            return "Vertical Motion"