from ml_logger import logger
import cv2
logger.configure("/alanyu/scratch/lucid_sim/stairs_v1")

# use cv2 to read a video and load the stack of frames
cap = cv2.VideoCapture("data/EI_Stairs_2_with_background.mp4")
frames = []
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frames.append(frame)

logger.save_video(frames, "nerf_stairs1.mp4", fps=30)
