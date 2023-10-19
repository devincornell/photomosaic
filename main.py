from __future__ import annotations
import pathlib
import skimage
import dataclasses
import typing
from PIL import Image
import numpy as np
import random
import proj
import tqdm
import multiprocessing
random.seed(0)


def find_best_thread(args: typing.Tuple[int, proj.SubCanvas]) -> proj.Canvas:
    ind, subtarget = args
    random.seed(ind)
    print(ind, 'starting')

    imman = proj.ImageManager.from_folders(
        source_folder=pathlib.Path("data/coco_train"),
        thumb_folder=pathlib.Path("data/coco_thumbs"),
        scale_res=subtarget.size,
        extensions=('png','jpg'),
    )

    best: proj.Canvas = None
    best_dist = float('inf')
    for j, si in tqdm.tqdm(enumerate(imman)):
        try:
            c = si.read_canvas()
            #return c # NOTE: TESTING ONLY. COMMENT OUT OTHERWISE
            d = subtarget.composite_dist(c)
        except Exception as e:
            print(e)
            print(si.source_fpath)
            exit()
        
        if d < best_dist:
            best_dist = d
            best = c

        if j % 1000 == 0:
            best.write_image(f'data/obama10x10_euclid/current_{ind}.png')
    best.write_image(f'data/obama10x10_euclid/best_{ind}.png')
    return best



if __name__ == "__main__":
    target = proj.Canvas.read_image(pathlib.Path("data/targets/obama.png"))
    
    height, width = 10, 10
    subtargets = list(enumerate(target.split_subcanvases(width, height)))
    with multiprocessing.Pool(9) as pool:
        best: typing.List[proj.Canvas] = list(pool.map(find_best_thread, subtargets))
    
    canvas_final = proj.Canvas.from_subcanvases(best, width)
    canvas_final.write_image(f'data/obama{height}x{width}_euclid.png')

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






