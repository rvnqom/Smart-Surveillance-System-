import cv2

class MotionDetector:
    def __init__(self, min_area=500):
        self.fgbg = cv2.createBackgroundSubtractorMOG2()
        self.min_area = min_area

    def detect(self, frame):
        fgmask = self.fgbg.apply(frame)
        contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        motion_detected = False
        for contour in contours:
            if cv2.contourArea(contour) > self.min_area:
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
                motion_detected = True

        return motion_detected, frame