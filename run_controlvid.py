from time import sleep

import jaynes
import inference
import tempfile

machines = [
    dict(ip="visiongpu50", gpu_id=0),
    dict(ip="visiongpu50", gpu_id=1),
    dict(ip="visiongpu50", gpu_id=2),
    dict(ip="visiongpu50", gpu_id=3),
    dict(ip="visiongpu50", gpu_id=4),
    dict(ip="visiongpu50", gpu_id=5),
    dict(ip="visiongpu50", gpu_id=6),
    dict(ip="visiongpu50", gpu_id=7),

    dict(ip="visiongpu54", gpu_id=0),
    dict(ip="visiongpu54", gpu_id=1),
    dict(ip="visiongpu54", gpu_id=2),
    dict(ip="visiongpu54", gpu_id=3),
    dict(ip="visiongpu54", gpu_id=4),
    dict(ip="visiongpu54", gpu_id=5),
    dict(ip="visiongpu54", gpu_id=6),
    dict(ip="visiongpu54", gpu_id=7),

    dict(ip="visiongpu55", gpu_id=0),
    dict(ip="visiongpu55", gpu_id=1),
    dict(ip="visiongpu55", gpu_id=2),
    dict(ip="visiongpu55", gpu_id=3),
]


def entrypoint(traj_num, env_type, vid_path, prefix):

    print(f"Hey guys! We're on host {traj_num} running environment type {env_type}")
    from ml_logger import logger

    print(f"RUN prefix {prefix} ")
    logger.configure(prefix=prefix)

    with tempfile.TemporaryDirectory() as cache_dir:
        logger.download_file(vid_path, to="temp_input.mp4")
        inference.main(traj_num=traj_num, env_type=env_type, vid_path=vid_path)
        print(f"Completed inference of trajectory {traj_num}, env type {env_type}")


if __name__ == "__main__":
    from ml_logger import logger

    LOGGER_PREFIX = "alanyu/scratch/lucid_sim/stairs_v1"
    logger.configure(prefix=LOGGER_PREFIX)

    filtered_videos = logger.glob("edges_ego_**.mp4")
    filtered_videos.sort()

    terrains = logger.load_json("terrains.json")
    runs = list(zip(terrains, filtered_videos))

    input(
        f"Running the following {len(runs)} configurations: {runs} \n Press enter to continue..."
    )

    for traj_num, (env_type, vid_name) in enumerate(runs, start=1):
        if traj_num < len(machines):
            host = machines[traj_num]["ip"]
            visible_devices = f'{machines[traj_num]["gpu_id"]}'
            jaynes.config(
                launch=dict(ip=host),
                runner=dict(gpus=f"'\"device={visible_devices}\"'"),
            )

            print(f"Setting up config {traj_num} on machine {host}")

            jaynes.run(
                entrypoint, traj_num=traj_num, env_type=env_type, vid_path=vid_name, prefix=LOGGER_PREFIX
            )
        sleep(2)

    jaynes.listen(200)
