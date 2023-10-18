from __future__ import annotations
import pathlib
import dataclasses
import typing
from PIL import Image
import numpy as np
import random
import proj

random.seed(0)

if __name__ == "__main__":
    imman = proj.ImageManager.from_folders(
        target_fpath=pathlib.Path("data/targets/black_circle2.png"),
        source_folder=pathlib.Path("data/test_original"),
        extensions=('png',),
    )
    target = imman.target_canvas()

    canvas_sample = imman.random_canvases(10)
    for i, cv in enumerate(canvas_sample):
        print(cv.dist(target))

