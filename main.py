from __future__ import annotations
import pathlib
import skimage
import dataclasses
import typing
from PIL import Image
import numpy as np
import random
import tqdm
import multiprocessing
random.seed(0)


import canvas

def find_best_thread(args: typing.Tuple[int, canvas.SubCanvas]) -> canvas.Canvas:
    ind, subtarget = args
    random.seed(ind)
    print(ind, 'starting')

    imman = canvas.ImageManager.from_folders(
        source_folder=pathlib.Path("data/coco_train"),
        thumb_folder=pathlib.Path("data/coco_thumbs"),
        scale_res=subtarget.size,
        extensions=('png','jpg'),
    )
    op = pathlib.Path("data/test/")
    op.mkdir(exist_ok=True, parents=True)

    best: canvas.Canvas = None
    best_dist = float('inf')
    source_images = imman.sample_source_images(100000)
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

        #if j % 10000 == 0:
        #    best.write_image(op.joinpath(f'current_{ind}.png'))
    #best.write_image(op.joinpath(f'best_{ind}.png'))
    return best



def find_best_chunk_thread(args) -> canvas.SubCanvasScores:
    thread_index: int = args[0]
    width: int = args[1]
    source_images: typing.List[canvas.SourceImage] = args[2]
    target_subcanvases: typing.List[canvas.SubCanvas] = args[3]
    outfolder: pathlib.Path = args[4]
    
    # compute distances for every source image to every target subcanvas
    distances: typing.List[typing.Tuple[int,float,canvas.SubCanvas]] = list()
    for si in tqdm.tqdm(source_images):
        try:
            si_canvas = si.retrieve_canvas()
            for ind, target_sc in enumerate(target_subcanvases):
                d = target_sc.dist.composit(si_canvas)
                distances.append((ind, d, si_canvas))
        except ValueError as e:
            print(f'\ncouldn\'t load image {si.source_fpath}')
    scs = canvas.SubCanvasScores.from_distances(distances)
    best = scs.to_canvas(width)
    best.write_image(outfolder.joinpath(f'current_{thread_index}.png'))
    print(f'\nfinished {thread_index}')
    return scs

if __name__ == "__main__":
    target = canvas.Canvas.read_image(pathlib.Path("data/targets/obama_bigger.jpeg"))
    print(target.im.dtype, target.im.shape)
    outfolder = pathlib.Path("data/obama_bigger_20x20-01/")
    outfolder.mkdir(exist_ok=True, parents=True)
    
    if True:
        height, width = 20, 20
        subtargets = list(target.split_subcanvases(height, width))

        imman = canvas.ImageManager.from_folders(
            source_folder=pathlib.Path("/StorageDrive/unzipped_photos/Takeout/main_photos/"),
            thumb_folder=pathlib.Path("data/personal_thumbs/"),
            scale_res=subtargets[0].size,
            extensions=('png','jpg', 'JPG'),
        )
        print(f'{len(imman)=}')
        #exit()
        batches = [(i,width,bi,subtargets, outfolder) for i,bi in enumerate(imman.chunk_source_images(height * width * 2))]
        
        print(f'running {len(batches)} batches against {len(subtargets)} subcanvases')
        with multiprocessing.Pool(12) as pool:
            map_func = pool.map
            scss: typing.List[canvas.SubCanvasScores] = list(map_func(find_best_chunk_thread, batches))

        best = scss[0]
        for i in range(1,len(scss)):
            best = best.reduce_subcanvasscores(scss[i])
        best.to_canvas(width).write_image(outfolder.joinpath(f'final.png'))
        
    
    if False:
        height, width = 20, 32
        subtargets = list(enumerate(target.split_subcanvases(height, width)))
        with multiprocessing.Pool(9) as pool:
            map_func = pool.map
            best: typing.List[canvas.Canvas] = list(map_func(find_best_thread, subtargets))
        
        canvas_final = canvas.Canvas.from_subcanvases(best, width)
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






