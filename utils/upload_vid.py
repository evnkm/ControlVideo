from ml_logger import logger
import cv2
from PIL import Image, ImageSequence

logger.configure("/evan_kim/scratch/lucid_sim/openpose")


def mp4_to_logger(file_path, logger_file_path):
    cap = cv2.VideoCapture(file_path)
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    logger.save_video(frames, logger_file_path, fps=30)


if __name__ == "__main__":
    mp4_to_logger("/home/evan/ControlVideo/data/shuffle.mp4", "shuffle.mp4")
