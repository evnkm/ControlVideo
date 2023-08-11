import random

STRUCTURE_TYPES = {
    "flat",
    "rough",
    "stairs_up",
    "stairs_up",
    "stairs_down",
    "stairs_down",
    "stairs_up_down",
    "stairs_up_down",
    "pyramid",
    "inverted_pyramid",
}

TASK_TYPES = {
    "ball",
}


class Adj:
    """
    Adjectives for prompts:
    color, sky, material, texture, ground, lighting
    """
    color = [
        "gray color",
        "beige color",
        "khaki color",
        "brown color",
        "maroon color",
        "pink color",
        "plum color",
        "lavender color",
        "indigo color",
        "azure color",
        "teal color",
        "green color",
        "pale yellow color",
        "orange color",
        "silver color",
    ]

    sky = [
        "blue sky",
        "grey sky",
        "cloudy",
        "sunlight",
        "no sun",
    ]

    material = [
        "concrete material",
        "wood material",
        "metal material",
        "plastic material",
        "rubber material",
        "glass material",
        "ceramic material",
        "carpet material",
        "sandstone material",
        "marble material",
        "granite material",
        "brick material",
        "asphalt material",
    ]

    texture = [
        "smooth",
        "rough",
        "bumpy",
        "granular",
    ]

    ground = [
        "gravel",
        "sand",
        "dirt",
        "grass",
        "mud",
        "snow",
        "pavement",
    ]

    lighting = [
        "dim",
        "dark",
        "shadows",
        "shadowy",
        "neutral lighting",
        "natural lighting",
        "bright",
    ]

    backdrop = [
        "trees",
        "buildings",
        "mountains",
        "city streets",
    ]


def prompt_gen(env_type: str):
    # structures like stairs, pyramids, etc.
    if env_type in STRUCTURE_TYPES:
        prompt_pfx = f"walking over {env_type}, first-person view, "
        if "stair" in env_type or "pyramid" in env_type:
            prompt_pfx = prompt_pfx + "sharp stair edges, "

        prompt = ", ".join(
            [
                random.choice(Adj.color),
                random.choice(Adj.sky),
                random.choice(Adj.material),
                random.choice(Adj.texture),
                random.choice(Adj.lighting),
            ]
        )

    # tasks such as ball following
    elif env_type in TASK_TYPES:
        # TODO: may need to generalize for other tasks
        if random.randint(0, 1) == 0:  # 50% chance of natural ground
            prompt_pfx = f"following a {random.choice(Adj.color)} ball, first-person view,"
            prompt = ", ".join(
                [
                    random.choice(Adj.sky),
                    random.choice(Adj.texture),
                    random.choice(Adj.ground),
                    random.choice(Adj.lighting),
                    random.choice(Adj.backdrop),
                ]
            )
        else:  # 50% chance of material ground
            prompt_pfx = f"following a {random.choice(Adj.color)} ball, first-person view, " \
                         f"{random.choice(Adj.material)} floor, "
            prompt = ", ".join(
                [
                    random.choice(Adj.sky),
                    random.choice(Adj.texture),
                    random.choice(Adj.lighting),
                    random.choice(Adj.backdrop),
                ]
            )

    return prompt_pfx + prompt
