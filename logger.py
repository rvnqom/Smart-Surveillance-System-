class Logger:
    def __init__(self, log_file="outputs/detections.log"):
        self.log_file = log_file

    def log(self, message):
        with open(self.log_file, "a") as log:
            log.write(message + "\n")