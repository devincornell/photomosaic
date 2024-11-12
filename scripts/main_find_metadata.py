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
import collections

random.seed(0)


import canvas

if __name__ == "__main__":
    manager = canvas.ImageFileManager(
        root_path = pathlib.Path('/StorageDrive/unzipped_photos/Takeout/'),
    )
    sources = manager.read_files(extensions=('png','jpg'))
    print(f'read {len(sources)=}')
    
    cts = collections.Counter([str(si.get_source_md()) for si in sources])
    #all_sources = set()
    #for si in sources:
    #    all_sources.add(str(si.get_source_md()))
    
    for smd, ct in cts.items():
        print(f'{ct}=, {smd=}')
    print(len([s for s in sources if s.is_usable()]))
    
    
    