import random

ALLOWED_TYPES = {
    "flat",
    "rough",
    "stairs_up",
    "stairs_down",
    "pyramid",
    "inverse_pyramid",
}

colors = [
    "light red",
    "pink",
    "pale green",
    "green",
    "light blue",
    "light grey",
    "dark grey",
    "tan",
    "brown",
    "pale yellow",
]

sky = [
    "blue sky",
    "grey sky",
    "cloudy",
    "sunlight",
    "no sun",
]

material = [
    "concrete",
    "wood",
    "metal",
    "plastic",
    "rubber",
    "glass",
    "ceramic",
    "carpet",
    "sandstone",
    "marble",
    "granite",
    "brick",
    "stone",
]

texture = [
    "smooth",
    "rough",
    "bumpy",
    "granular",
    "gravel",
    "sandy",
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


def prompt_gen():
    prompt = ', '.join(
        [random.choice(colors),
         random.choice(sky),
         random.choice(material),
         random.choice(texture),
         random.choice(lighting)])
    return prompt
