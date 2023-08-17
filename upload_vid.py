from ml_logger import logger
import cv2
logger.configure("/evan_kim/scratch/lucid_sim/openpose")

# use cv2 to read a video and load the stack of frames
cap = cv2.VideoCapture("data/walk1.mp4")
frames = []
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frames.append(frame)

logger.save_video(frames, "walk1.mp4", fps=30)
