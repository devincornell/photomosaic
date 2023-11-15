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
import tqdm
import os
import coproc

import coproc
import doctable

import canvas

def new_monitor(outfolder: pathlib.Path) -> coproc.Monitor:
    return coproc.Monitor(
        fig_path=outfolder.joinpath('best_progress.png'), 
        log_path=outfolder.joinpath('best_progress.log'), 
        snapshot_seconds=1, 
        save_fig_freq=1
    )

def main(preprocess_thumbs: bool = False):
    outfolder = pathlib.Path("data/experimental/")
    
    with new_monitor(outfolder) as monitor:
        db = canvas.DistanceDB.open(outfolder.joinpath('dists/distances.db'))
        print(db.q.count_by_target().df())
        #print(db.q.select_top_distances(10).df().columns)
        print(db.q.select_top_distances(10)[0])
        
        
if __name__ == "__main__":
    main()




