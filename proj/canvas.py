from __future__ import annotations
import pathlib
import dataclasses
import typing
from PIL import Image, ImageChops
import skimage
import numpy as np
import random

from .util import imread_transform, write_as_uint

@dataclasses.dataclass
class Distances:
    canvas: CanvasBase
    def composit(self, other: CanvasBase) -> float:
        return self.euclid(other) + self.sobel(other)

    def euclid(self, other: CanvasBase) -> float:
        return np.linalg.norm(self.canvas.im - other.im)
            
    def sobel(self, other: CanvasBase) -> float:
        return np.linalg.norm(self.canvas.transform_sobel() - other.transform_sobel())

@dataclasses.dataclass
class CanvasBase:
    im: np.ndarray
    
    @property
    def dist(self) -> Distances:
        return Distances(self)
    
    @property
    def size(self) -> typing.Tuple[int,int]:
        '''Height, width.'''
        return self.im.shape[:2]
            
    def write_image(self, fpath: pathlib.Path) -> np.ndarray[np.uint16]:
        return write_as_uint(self.im, fpath)
    
    def to_pillow(self) -> Image.Image:
        return Image.fromarray(self.im)
    
    def transform_sobel(self) -> np.ndarray:
        return skimage.filters.sobel(self.im)

@dataclasses.dataclass
class Canvas(CanvasBase):
    fpath: pathlib.Path

    @classmethod
    def read_image(cls, fpath: pathlib.Path) -> Canvas:
        return cls(
            fpath=fpath,
            im=imread_transform(fpath),
        )
    
    def split_subcanvases(self, x_divisions: int, y_divisions: int) -> typing.List[SubCanvas]:
        h,w = self.size
        x_size = w // x_divisions
        y_size = h // y_divisions
        subcanvases = []
        for iy in range(y_divisions):
            for ix in range(x_divisions):
                sc = SubCanvas.from_canvas(self, ix*x_size, iy*y_size, x_size, y_size)
                subcanvases.append(sc)
        return subcanvases

    @classmethod
    def from_subcanvases(cls, scanvases: typing.List[CanvasBase], width: int) -> Canvas:
        '''Reconstruct a canvas from a list of subcanvases.'''
        ch,cw = scanvases[0].size
        full_w = width * cw
        full_h = (len(scanvases) // width) * ch
        im = np.zeros((full_h,full_w,3), dtype=np.float64)
        for i,sc in enumerate(scanvases):
            grid_y, grid_x = i // width, i % width
            im[grid_y*ch:(grid_y+1)*ch, grid_x*cw:(grid_x+1)*cw, :] = sc.im
        return cls(
            fpath=None,
            im=im,
        )

    
@dataclasses.dataclass
class SubCanvas(CanvasBase):
    canvas: Canvas

    @classmethod
    def from_canvas(cls, canvas: Canvas, x: int, y: int, w: int, h: int) -> SubCanvas:
        im = canvas.im[y:y+h,x:x+w]
        return cls(
            canvas=canvas,
            im=im,
        )



