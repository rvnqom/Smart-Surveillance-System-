import os

def voice_alert(message="motion detected"):
    os.system(f'say "{message}"')  # Works on Mac