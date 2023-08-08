import tempfile
import inference


def test_run():
    from ml_logger import logger
    # logger.configure(prefix="/lucid_sim/datasets/lucid_sin/test_main")
    logger.configure(prefix="/alanyu/scratch/lucid_sim/stairs_v1")
    to_video = "edges_ego_0001.mp4"

    # Create a temporary cache file
    with tempfile.TemporaryDirectory() as cache_dir:
        logger.download_file(to_video, to="temp_input.mp4")
        inference.main(env_type="flat", vid_path="temp_input.mp4", traj_num=1)
        print("finished inference")

test_run()
