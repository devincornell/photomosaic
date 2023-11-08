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

    def copy(self, **new_values) -> Image:
        '''Copy all but provided attributes.'''
        return self.__class__(**{**dataclasses.asdict(self), **new_values})
    
    def cast(self, new_type: type, **additional_data) -> Image:
        return new_type(**{**dataclasses.asdict(self), **additional_data})

    ################ Dunder ################
    def __getitem__(self, ind: typing.Union[slice, typing.Tuple[slice, ...]]) -> Image:
        '''Get image at index or (y,x) index.'''
        return self.copy(im=self.im[ind])
    
    def slice(self, y: int, x: int, h: int, w: int) -> Image:
        '''Get image at index or (y,x) index.'''
        return self[x:x+w, y:y+h]

    ################ Properties ################
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



