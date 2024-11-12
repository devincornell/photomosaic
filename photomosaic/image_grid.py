from __future__ import annotations
#import pathlib
from pathlib import Path
import skimage
import dataclasses
import typing
import numpy as np

import mediatools

GridIndex = int

@dataclasses.dataclass
class ImageGrid:
    '''A grid of subimages.'''
    subimages: list[mediatools.Image|None]
    x_slices: int
    y_slices: int
    
    @classmethod
    def from_image(cls, im: mediatools.Image, x_slices: int, y_slices: int) -> typing.Self:
        '''From an image, create a grid of subimages.'''
        h, w = im.size
        y_slice_size, x_slice_size = h // y_slices, w // x_slices
        
        subimages: list[mediatools.Image] = list()
        for y in range(y_slices):
            for x in range(x_slices):
                subimages.append(im[y*y_slice_size:(y+1)*y_slice_size, x*x_slice_size:(x+1)*x_slice_size])
        return cls(
            subimages=subimages,
            x_slices=x_slices,
            y_slices=y_slices,
        )
    
    def recombine(self) -> mediatools.Image:
        '''Recombine the subimages into a single image.'''
        h, w = self.subimage_size
        im = np.zeros((h*self.y_slices, w*self.x_slices, 3), dtype=np.float64)
        for i, subimage in enumerate(self.subimages):
            if subimage is not None:
                y, x = i // self.x_slices, i % self.x_slices
                im[y*h:(y+1)*h, x*w:(x+1)*w] = subimage.im
        return mediatools.Image(im)

    @property
    def subimage_size(self):
        for si in self.subimages:
            if si is not None:
                return si.size
        raise ValueError('No subimages found in grid.')

    def calc_distances(self, cand_img: mediatools.Image) -> list[tuple[float, GridIndex]]:
        '''Calculate distances between each subimage and a candidate image.'''
        return [(si.dist.composit(cand_img), i) for i, si in enumerate(self.subimages)]

    def __getitem__(self, i: GridIndex | slice) -> mediatools.Image | list[mediatools.Image]:
        '''Slice into subimage list.'''
        return self.subimages[i]
    
    def __len__(self):
        return len(self.subimages)
    
    def __iter__(self):
        return iter(self.subimages)

