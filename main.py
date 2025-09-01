import cv2
import torch
import torch.nn as nn
import torchvision
from motion_detection import MotionDetector
from recorder import Recorder
from logger import Logger
from alert import voice_alert

# -----------------------------
# Config
# -----------------------------
frames_per_clip = 8
device = "mps" if torch.backends.mps.is_available() else "cpu"

# -----------------------------
# Initialize components
# -----------------------------
detector = MotionDetector(min_area=2500)  # ignore subtle movements
recorder = Recorder()
logger = Logger()

# -----------------------------
# Load trained action recognition model
# -----------------------------
num_classes = 2
feature_model = torchvision.models.video.r3d_18(weights=None)  # no pretrained weights
feature_model.fc = nn.Linear(512, num_classes)  # replace final layer
feature_model.load_state_dict(torch.load("r3d18_demo_best.pth", map_location=device))
feature_model.to(device)
feature_model.eval()

class_names = ["walking", "running"]
clip_frames = []

# -----------------------------
# OpenCV capture
# -----------------------------
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_display = frame.copy()

    # ----- Motion detection -----
    motion, frame_with_boxes = detector.detect(frame)

    if motion:
        clip_frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        if len(clip_frames) == frames_per_clip:
            # Preprocess frames
            processed = []
            for f in clip_frames:
                f = cv2.resize(f, (80, 80))
                f = torch.tensor(f).permute(2, 0, 1).float() / 255.0
                mean = torch.tensor([0.43216, 0.394666, 0.37645]).view(3, 1, 1)
                std = torch.tensor([0.22803, 0.22145, 0.216989]).view(3, 1, 1)
                f = (f - mean) / std
                processed.append(f)

            clip_tensor = torch.stack(processed, dim=1).unsqueeze(0).to(device)

            # Prediction
            with torch.no_grad():
                output = feature_model(clip_tensor)
                pred_class = output.argmax(dim=1).item()

            action = class_names[pred_class]

            # Display on video
            cv2.putText(
                frame_display,
                f"Action: {action}",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
            )

            # -----------------------------
            # Action-specific handling
            # -----------------------------
            if action == "walking":
                logger.log("Walking detected")  # log only
                if recorder.recording:  # stop if it was recording
                    recorder.stop()

            elif action == "running":
                logger.log("Running detected")
                voice_alert("Running detected")  # alert
                if not recorder.recording:
                    recorder.start(frame)
                recorder.write(frame)  # record

            clip_frames = []

    else:
        if recorder.recording:
            recorder.stop()

    # Display
    cv2.imshow("Smart Surveillance", frame_display)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC key to quit
        break

cap.release()
cv2.destroyAllWindows()