from __future__ import annotations
import dataclasses
import typing
import skimage
import numpy as np
import pathlib

from .imagegrid import ImageGrid
from .distances import Distances

class Height(int):
    pass

class Width(int):
    pass

@dataclasses.dataclass(frozen=True)
class Image:
    im: np.ndarray

    @property
    def dist(self) -> Distances:
        return Distances(self)

    @property
    def size(self) -> typing.Tuple[Height, Width]:
        '''Height, width.'''
        return self.im.shape[:2]
    
    @property
    def shape(self) -> typing.Tuple[Height, Width, int]:
        '''Shape of image.'''
        return self.im.shape

    def copy(self, **new_values) -> Image:
        '''Copy all but provided attributes.'''
        return self.__class__(**{**dataclasses.asdict(self), **new_values})

    ################ Read/Writing ################
    @classmethod
    def read(cls, path: pathlib.Path) -> Image:
        return cls.copy(im = skimage.io.imread(str(path)))

    def write_ubyte(self, path: pathlib.Path) -> None:
        '''Writes image as uint8.'''
        return self.as_ubyte().write(path)
    
    def write(self, path: pathlib.Path) -> None:
        '''Writes image as float.'''
        return skimage.io.imsave(str(path), self.im)

    ################ Transforms ################

    def sobel(self) -> Image:
        return self.copy(im=skimage.filters.sobel(self.im))
    
    def resize(self, resize_res: typing.Tuple[int,int]) -> Image:
        return self.copy(im=skimage.transform.resize(self.im, resize_res))

    def transform_color_rgb(self) -> np.ndarray:
        '''Transform image to be rgb.'''
        if len(self.im.shape) < 3:
            im = skimage.color.gray2rgb(self.im)
        elif im.shape[2] > 3:
            im = skimage.color.rgba2rgb(self.im)
        return self.copy(im=im)

    
    ################ Conversions ################
    def as_ubyte(self) -> Image:
        return self.copy(im=skimage.img_as_ubyte(self.im))
    
    def as_float(self) -> Image:
        return self.copy(im=skimage.img_as_float(self.im))

    ################ Splitting and Recombining ################

    def split_grid(self, y_divisions: int, x_divisions: int) -> typing.List[Image]:
        '''Create by dividing up a single canvas into equal parts.'''
        h,w = self.size
        x_size = w // x_divisions
        y_size = h // y_divisions
        subimages = []
        for iy in range(y_divisions):
            for ix in range(x_divisions):
                sc = self.__class__.slice(self, ix*x_size, iy*y_size, x_size, y_size)
                subimages.append(sc)
        return subimages

    @classmethod
    def from_image_grid(cls, image_grid: ImageGrid, width: int) -> Canvas:
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


