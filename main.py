from __future__ import annotations
#import pathlib
from pathlib import Path
import skimage
import dataclasses
import typing
from PIL import Image
import numpy as np
import random
import tqdm
import multiprocessing

import mediatools
import photomosaic


def transform_img(img: mediatools.Image, size: tuple[int,int]) -> mediatools.Image:
    '''Transform an image to a comparable format.'''
    return img.as_float().to_rgb().transform.resize(size)

def greedy_optimize_thread(
    thread_index: int, 
    cand_files: list[mediatools.ImageFile], 
    sgrid: photomosaic.ImageGrid, 
    outfolder: Path
) -> photomosaic.GridScores:
    best_scores = photomosaic.GridScores.from_grid_size(x_slices=sgrid.x_slices, y_slices=sgrid.y_slices)
    for i, cand_file in tqdm.tqdm(enumerate(cand_files), ncols=100, total=len(cand_files)):

        # transform image to comparable format
        cand_img = transform_img(cand_file.read(), size=sgrid.subimage_size)

        # compute distances between candidate and all subimages
        dists = sgrid.calc_distances(cand_img)
        
        # insert the candidate image if it is the best for any subimage
        best_scores.insert_any_if_best(cand_img, dists)

        # save this every iteration so we can see it happen
        if i % 5 == 0:
            best_score_grid = best_scores.get_subimage_grid()
            best_score_grid.recombine().as_ubyte().write(outfolder.joinpath(f'current_{thread_index}.png'))
    return best_scores


def main():
    
    output_fname = 'data/tmp/best_score_grid_obama.png'

    base_file = mediatools.ImageFile.from_path("data/targets/obama.png")
    base_image = base_file.read().to_rgb().transform.resize((500, 500))
    sgrid = photomosaic.ImageGrid.from_image(base_image, 10, 10)
    print(base_image)

    cand_files = mediatools.ImageFiles.from_rglob('data/dataset_coco/train/')
    print(len(cand_files))

    # all the work happens here!
    best_scores = greedy_optimize_thread(0, cand_files[:None], sgrid, Path('data/tmp/'))
    
    best_score_grid = best_scores.get_subimage_grid()
    best_score_grid.recombine().as_ubyte().write(output_fname)

if __name__ == "__main__":
    main()




