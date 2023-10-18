from __future__ import annotations
import pathlib
import dataclasses
import typing
from PIL import Image
import numpy as np
import random
import proj
import tqdm

random.seed(0)

def find_best(target_subcanvas: proj.SubCanvas, canvases: typing.List[proj.Canvas]) -> tuple[proj.Canvas, float]:
    dists = [cv.composite_dist(target_subcanvas) for cv in canvases]
    ind = np.argmin(dists)
    return canvases[ind], dists[ind]

if __name__ == "__main__":
    target = proj.Canvas.read_image(pathlib.Path("data/targets/black_circle1.png"))
    subtargets = target.split_subcanvases(3,3)

    imman = proj.ImageManager.from_folders(
        source_folder=pathlib.Path("data/coco_train"),
        thumb_folder=pathlib.Path("data/coco_thumbs"),
        scale_res=subtargets[0].size,
        extensions=('png','jpg'),
    )
    print(len(imman))

    for i, st in enumerate(subtargets):
        best: proj.Canvas = None
        best_dist = float('inf')
        for batch in tqdm.tqdm(imman.batches(1000)):
            canvases = [si.read_canvas() for si in tqdm.tqdm(batch)]
            
            bc, d = find_best(st, canvases)
            if d < best_dist:
                best_dist = d
                best = bc
        best.write_image(f'data/best{i}.png')

    exit()
    exit()
    canvas_sample = imman.random_canvases(1000)
    #bc, d = find_best(subtargets[0], canvas_sample)
    subtargets[0].write_image('data/target.png')
    current_best = float('inf')
    for cs in tqdm.tqdm(canvas_sample):
        d = cs.sobel_dist(subtargets[0])
        if d < current_best:
            cs.write_image(f'data/best.png')
            current_best = d
