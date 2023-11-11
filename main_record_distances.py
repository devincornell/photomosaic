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
import coproc

random.seed(0)

import canvas


import tqdm
import os

def parallel_batch_and_calc_distances(
    imman: canvas.ImageManager, 
    grid: canvas.ImageGrid, 
    pickle_path: pathlib.Path,
    monitor: coproc.MonitorMessengerInterface,
    batch_size: int = 10,
    processes: int = os.cpu_count(),
) -> None:
    pickle_path.mkdir(exist_ok=True, parents=True)
    
    monitor.print(f'batching into sizes of {batch_size}')
    imman_batches = imman.batch_image_managers(batch_size=batch_size)
    monitor.print(f'{len(imman_batches)=}')
    
    with multiprocessing.Pool(processes) as pool:
        monitor.print(f'started {processes} processes')
        
        monitor.update_child_processes()
        
        map_func = pool.imap_unordered
        batch_it = [(imman, grid) for imman in (imman_batches)]
        
        monitor.print('starting main loop')
        for i, r in tqdm.tqdm(enumerate(map_func(thread_calc_and_save_distances, batch_it)), total=len(batch_it)):
            monitor.label(f'finished batch {i}')
            
            with pickle_path.joinpath(f'batch_{i}.pkl').open('wb') as f:
                pickle.dump(r, f)
            
            monitor.label(f'saved pickle {pickle_path}')
    
import pickle
def thread_calc_and_save_distances(args):
    imman: canvas.ImageManager = args[0]
    grid: canvas.ImageGrid = args[1]
    
    dists = dict()
    for thumb in imman.read_thumbs():
        for (y,x), si in grid.images():
            #print('.', end='', flush=True)
            d = thumb.dist.composit(si)
            dists[(str(thumb.path), (y,x))] = d
    return dists




import coproc

def new_monitor(outfolder: pathlib.Path) -> coproc.Monitor:
    return coproc.Monitor(
        fig_path=outfolder.joinpath('progress.png'), 
        log_path=outfolder.joinpath('progress.log'), 
        snapshot_seconds=1, 
        save_fig_freq=1
    )

def main(preprocess_thumbs: bool = False):
    outfolder = pathlib.Path("data/experimental/")
    #outfolder.mkdir(exist_ok=True, parents=True)
    
    with new_monitor(outfolder) as monitor:
        monitor.add_note('reading and resizing target')
        thumb_res = (64, 64)
        height_images = 10
        target = canvas.FileImage.read(
            #path=pathlib.Path("data/targets/Gray-Mountain_small.webp"),
            path=pathlib.Path("data/targets/lofi_tiny.jpg"),
        )
        h,w = target.size
        aspect = h / w
        new_height = height_images * thumb_res[0]
        new_width = int(new_height / aspect)
        target = target.resize((new_height, new_width))
        monitor.print(f'{target.size=}')
        
        grid = canvas.ImageGrid.from_fixed_subimages(target, thumb_res[0], thumb_res[1])
        monitor.print(f'grid size: {grid.size=}')
        monitor.print(f'subimage: {grid[0,0].size=}, {grid.sub_size=}')
        monitor.print(f'full image: {grid.to_image().size=}, {grid.full_size=}')
        
                
        monitor.label('grabbing source images')
        imman = canvas.ImageManager.from_rglob(
            source_folder=pathlib.Path("/StorageDrive/unzipped_photos/Takeout/"),
            #source_folder=pathlib.Path("/StorageDrive/unzipped_photos/Takeout/Google Photos/V_D/"),
            thumb_folder=pathlib.Path("data/personal_thumbs2/"),
            scale_res=thumb_res,
            extensions=('png','jpg', 'JPG'),
        )
        
        monitor.print(f'{len(imman)=}')
        imman = imman.filter_usable_photo(use_tqdm=True)
        monitor.print(f'{len(imman)=} (after filtering)')
        
        #imman = imman.clone(source_images=imman.source_images[:1000])
        
        monitor.print(f'preprocessing {len(imman)=} thumbs')
        for t in imman.read_thumbs_parallel(use_tqdm=True, processes=4):
            #monitor.update_child_processes()
            pass
        monitor.print(f'finished.')
        
        parallel_batch_and_calc_distances(
            imman=imman,
            grid=grid,
            pickle_path=outfolder.joinpath('dists/'),
            monitor = monitor,
            batch_size=100,
            processes=4,
        )
        
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




