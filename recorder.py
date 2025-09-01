import cv2
import datetime

class Recorder:
    def __init__(self, output_dir="outputs"):
        self.fourcc = cv2.VideoWriter_fourcc(*'XVID')
        self.out = None
        self.recording = False
        self.output_dir = output_dir

    def start(self, frame):
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.out = cv2.VideoWriter(
            f"{self.output_dir}/motion_{timestamp}.avi",
            self.fourcc,
            20.0,
            (frame.shape[1], frame.shape[0])
        )
        self.recording = True
        return timestamp

    def write(self, frame):
        if self.recording and self.out:
            self.out.write(frame)

    def stop(self):
        if self.out:
            self.out.release()
        self.recording = False