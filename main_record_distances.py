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
        except OSError as e:
            print(f'\ncouldn\'t load image {si.source_fpath}')
    scs = canvas.SubCanvasScores.from_distances(distances)
    best = scs.to_canvas(width)
    best.write_image(outfolder.joinpath(f'current_{thread_index}.png'))
    print(f'\nfinished {thread_index}')
    return scs

def main():
    import coproc


    thumb_res = (67, 89)
    height_images = 30
    target = canvas.FileImage.read(
        #path=pathlib.Path("data/targets/Gray-Mountain_small.webp"),
        path=pathlib.Path("data/targets/lofi_tiny.jpg"),
    )
    h,w = target.size
    aspect = h / w
    new_height = height_images * thumb_res[0]
    new_width = int(new_height / aspect)
    target = target.resize((new_height, new_width))
    print(f'{target.size=}')
    
    grid = canvas.ImageGrid.from_fixed_subimages(target, thumb_res[0], thumb_res[1])
    print(f'{grid[0,0].size=}')
    #grid.to_image().as_ubyte().write('tmp.png')
    print(f'{grid.to_image().size=}')
    
    exit()
    height, width = 30, 30
    outfolder = pathlib.Path("data/lofi-30x30/")
    outfolder.mkdir(exist_ok=True, parents=True)

    the_monitor = coproc.Monitor(
        fig_path=outfolder.joinpath('progress.png'), 
        log_path=outfolder.joinpath('progress.log'), 
        snapshot_seconds=1, 
        save_fig_freq=1
    )

    with the_monitor as monitor:
        #monitor.add_note(target.im.dtype, target.im.shape, do_print=True)    
            
        #height, width = 30, 30
        #subtargets = list(target.split_subcanvases(height, width))
        
        
        thumb_res = (67, 89)
        imman = canvas.ImageManager.from_rglob(
            source_folder=pathlib.Path("/StorageDrive/unzipped_photos/Takeout/"),
            thumb_folder=pathlib.Path("data/personal_thumbs/"),
            scale_res=thumb_res,
            extensions=('png','jpg', 'JPG'),
        )
        monitor.add_note(f'{len(imman)=}', do_print=True)
        if True:
            imman = imman.filter_usable_photo(use_tqdm=True)
            monitor.add_note(f'{len(imman)=} (after filtering)', do_print=True)
            imman.create_thumbs(use_tqdm=True)
        exit()
        batches = [(i,width,bi,subtargets, outfolder) for i,bi in enumerate(imman.chunk_source_images(height * width * 2))]
        
        monitor.add_note(f'running {len(batches)} batches against {len(subtargets)} subcanvases')
        with multiprocessing.Pool(12) as pool:
            #monitor.update_child_processes()
            
            map_func = pool.imap_unordered
            scss: typing.List[canvas.SubCanvasScores] = list()
            for i, r in enumerate(map_func(find_best_chunk_thread, batches)):
                scss.append(r)
                monitor.add_note(f'finished batch {i}')
                
        monitor.add_note('finished all batches')
        best = scss[0]
        for i in range(1,len(scss)):
            best = best.reduce_subcanvasscores(scss[i])
        best.to_canvas(width).write_image(outfolder.joinpath(f'final.png'))
    

if __name__ == "__main__":
    main()




