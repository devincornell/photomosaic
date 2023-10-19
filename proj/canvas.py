from __future__ import annotations
import pathlib
import dataclasses
import typing
from PIL import Image, ImageChops
import skimage
import numpy as np
import random

@dataclasses.dataclass
class CanvasBase:
    im: np.ndarray

    @property
    def size(self) -> typing.Tuple[int,int]:
        '''Height, width.'''
        return self.im.shape[0], self.im.shape[1]
    
    def image(self) -> Image.Image:
        return Image.fromarray(self.im)
    
    def composite_dist(self, other: CanvasBase) -> float:
        #print(self.size, other.size, self.im.shape, other.im.shape)
        return self.euclid_dist(other) + self.sobel_dist(other)

    def euclid_dist(self, other: CanvasBase) -> float:
        return np.linalg.norm(self.im - other.im)
            
    def sobel_dist(self, other: CanvasBase) -> float:
        return np.linalg.norm(self.sobel() - other.sobel())

    def sobel(self) -> np.ndarray:
        return skimage.filters.sobel(self.im)
    
    def write_image(self, fpath: pathlib.Path) -> None:
        return skimage.io.imsave(str(fpath), skimage.img_as_ubyte(self.im))


@dataclasses.dataclass
class Canvas(CanvasBase):
    fpath: pathlib.Path

    @classmethod
    def read_image(cls, fpath: pathlib.Path) -> Canvas:
        im = cls.read_image_skimage(fpath)
        return cls(
            fpath=fpath,
            im=im,
        )
    
    @staticmethod
    def read_image_skimage(fpath: pathlib.Path) -> np.ndarray:
        im = skimage.io.imread(str(fpath))
        if len(im.shape) < 3:
            im = skimage.color.gray2rgb(im)
        elif im.shape[2] > 3:
            im = skimage.color.rgba2rgb(im)
        return im
    
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
        im = np.zeros((full_h,full_w,3), dtype=np.uint8)
        for i,sc in enumerate(scanvases):
            grid_y, grid_x = i // width, i % width

            print(i, grid_y, grid_x, sc.size, im.shape)
            print(grid_y*ch, (grid_y+1)*ch, grid_x*cw, (grid_x+1)*cw)
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



