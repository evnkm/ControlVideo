import tempfile
import inference

def one_trajectory():
    from ml_logger import logger
    # logger.configure(prefix="/lucid_sim/datasets/lucid_sin/test_main")
    logger.configure(prefix="/alanyu/scratch/lucid_sim/stairs_v1")
    to_video = "nerf_stairs1.mp4"
    env_type = "stairs"

    # Create a temporary cache file
    with tempfile.TemporaryDirectory() as cache_dir:
        logger.download_file(to_video, to="temp_input.mp4")
        inference.main(traj_num=4, env_type=env_type, vid_path="temp_input.mp4")
        print("finished inference")


one_trajectory()
