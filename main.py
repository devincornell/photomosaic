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
    op = pathlib.Path("data/test/")
    op.mkdir(exist_ok=True, parents=True)

    best: proj.Canvas = None
    best_dist = float('inf')
    source_images = imman.sample_source_images(10000)
    for j, si in tqdm.tqdm(enumerate(source_images)):
        try:
            c = si.retrieve_canvas()
            #print(c.im.dtype)
            #return c # NOTE: TESTING ONLY. COMMENT OUT OTHERWISE
            d = subtarget.dist.composit(c)
        except Exception as e:
            print(e)
            print(si.source_fpath)
            #exit()
            raise e
        
        if d < best_dist:
            best_dist = d
            best = c

        if j % 10000 == 0:
            best.write_image(op.joinpath(f'current_{ind}.png'))
    best.write_image(op.joinpath(f'best_{ind}.png'))
    return best

if __name__ == "__main__":
    target = proj.Canvas.read_image(pathlib.Path("data/targets/obama.png"))
    print(target.im.dtype, target.im.shape)
    
    height, width = 20, 32
    subtargets = list(enumerate(target.split_subcanvases(width, height)))
    with multiprocessing.Pool(9) as pool:
        map_func = pool.map
        best: typing.List[proj.Canvas] = list(map_func(find_best_thread, subtargets))
    
    canvas_final = proj.Canvas.from_subcanvases(best, width)
    canvas_final.write_image(f'data/obama{height}x{width}_composit.png')

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






