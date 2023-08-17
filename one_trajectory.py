import tempfile
import inference


def one_trajectory():
    from ml_logger import logger
    # logger.configure(prefix="/lucid_sim/datasets/lucid_sin/test_main")
    logger.configure(prefix="/evan_kim/scratch/lucid_sim/openpose")
    to_video = "walk1.mp4"
    # env_type = "stairs"

    # Create a temporary cache file
    with tempfile.TemporaryDirectory() as cache_dir:
        logger.download_file(to_video, to="temp_input.mp4")
        inference.simple("3 people dressed in hawaiian shirts walking dancing toward me on the beach, bright lighting, 4k", 1, 1, "temp_input.mp4")
        print("finished inference")


one_trajectory()
