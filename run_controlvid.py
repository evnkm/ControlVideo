from time import sleep

from itertools import product

import jaynes
import json
from params_proto import ParamsProto
from inference import Lucid, generate, main
import prompt_samples


machines = [
    # dict(ip="visiongpu50", gpu_id=0),
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


class RunArgs(ParamsProto):
    environments = ["Go1Terrain"]  # , "Anyma1Terrain"] # , "Go1"]  # , "Anyma1Terrain"]
    algos = ["PPO_Schulman"]
    seeds = [400, 500, 600]
    sweep = True


#
# checkpoints = [
#     "/alanyu/pql/investigation/Go1Terrain_no_delay/PPO_Schulman/400/checkpoints/model_last.pt",
#     "/alanyu/pql/investigation/Go1Terrain_no_delay/PPO_Schulman/500/checkpoints/model_last.pt",
#     "/alanyu/pql/investigation/Go1Terrain_no_delay/PPO_Schulman/600/checkpoints/model_last.pt",
# ]


def entrypoint(i,
               prompt,
               video_path,
               env_name,
               sample_root,
               sample_vid_name):
    print(f"Hey guys! We're on host {i} running environment {env}")
    from ml_logger import logger
    from ml_logger.job import RUN

    RUN.entity = "alanyu"
    RUN.project = "lucid_sim_runs"

    print(f" RUN prefix {prefix} ")
    logger.configure(prefix=prefix)

    # TODO: MAKE START INDEX CONSISTENT WITH ALAN
    generate(prompt, video_path, env_name, sample_root, sample_vid_name)


if __name__ == "__main__":
    from ml_logger import logger
    LOGGER_PREFIX = "alanyu/scratch/lucid_sim/stairs_v1"
    logger.configure(prefix=LOGGER_PREFIX)
    # TODO: generalize for other environments
    # environments = logger.glob("*")
    filtered_videos = logger.glob("edges_ego_*")

    # TODO: Update to read directly from app.dash.ml
    file_path = "data.json"
    with open(file_path, "r") as file:
        data_list = json.load(file)

    prompt_prefix = f"walking over {terrain_type} stairs, first-person view, sharp stair edges, "
    prompts = [prompt_prefix + prompt_samples.prompt_gen() for _ in range(len(filtered_videos))]

    input(
        f"Running the following {len(runs)} configurations: {runs} \n Press enter to continue..."
    )


        for

    prefix = LOGGER_PREFIX  # "/alanyu/pql/investigation_raw/"

    for i, (env, algo, seed) in enumerate(runs):
        if i < len(machines):
            host = machines[i]["ip"]
            visible_devices = f'{machines[i]["gpu_id"]}'
            jaynes.config(
                launch=dict(ip=host),
                runner=dict(gpus=f"'\"device={visible_devices}\"'"),
            )

            print(f"Setting up config {i} on machine {host}")
            # thunk = instr(entrypoint)

            jaynes.run(
                entrypoint, i=i, env=env, seed=seed, prefix=prefix + postfix
            )
        sleep(2)

    jaynes.listen(200)
