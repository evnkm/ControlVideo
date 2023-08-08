import numpy as np

from ml_logger import logger

outputs = np.random.rand(20, 64, 64, 3)
logger.save_video(outputs, "test.mp4", fps=50)