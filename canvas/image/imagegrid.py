from __future__ import annotations
import dataclasses
import typing
import skimage
import numpy as np
import pathlib

#if typing.TYPE_CHECKING:
from .image import Image, Height, Width
from .subimage import SubImage

class GridX(int):
    pass

class GridY(int):
    pass

@dataclasses.dataclass
class ImageGrid:
    subimages: typing.List[SubImage]
    y_divisions: int

    ################## factory method constructors ##################
    @classmethod
    def from_fixed_subimages(cls, image: Image, subimage_height: Height, subimage_width: Width) -> ImageGrid:
        '''Create by creating a grid of fixed-size sqares that fill the image.'''
        full_h,full_w = image.size
        return cls.from_specs(
            image = image,
            y_divisions = full_h // subimage_height,
            x_divisions = full_w // subimage_width,
            subimage_height = subimage_height,
            subimage_width = subimage_width,
        )
    
    @classmethod
    def from_even_divisions(cls, image: Image, y_divisions: int, x_divisions: int) -> typing.List[Image]:
        '''Create by dividing up a single canvas into equal parts.'''
        full_h,full_w = image.size
        return cls.from_specs(
            image = image,
            y_divisions = y_divisions,
            x_divisions = x_divisions,
            subimage_height = full_h // y_divisions,
            subimage_width = full_w // x_divisions,
        )
    
    @classmethod
    def from_specs(cls, image: Image, y_divisions: int, x_divisions: int, subimage_height: int, subimage_width: int) -> typing.List[SubImage]:
        '''Create a list of subimages that are empty.'''
        subimages = []
        for iy in range(y_divisions):
            for ix in range(x_divisions):
                subimages.append(SubImage.from_image(
                    image = image,
                    y = iy*subimage_height, 
                    x = ix*subimage_width, 
                    h = subimage_height, 
                    w = subimage_width,
                ))
        return cls(
            subimages = subimages, 
            y_divisions = y_divisions,
        )
    
    ################## Dunder ##################
    def __getitem__(self, ind: typing.Union[typing.Tuple[GridY,GridX],int]) -> Image:
        '''Get image at (y,x) location.'''
        y,x = ind
        if y >= self.y_divisions:
            raise IndexError(f'y index {y} is gt or eq to width {self.y_divisions}')
        return self.subimages[self.xy_to_ind(y,x,self.y_divisions)]

    ################## Grid mappings ##################
    @staticmethod
    def ind_to_xy(ind: int, y_divisions: int) -> typing.Tuple[GridY,GridX]:
        '''Converts an index to a grid location.'''
        return ind // y_divisions, ind % y_divisions
    
    @staticmethod
    def xy_to_ind(y: GridY, x: GridX, y_divisions) -> int:
        '''Converts a grid location to an index.'''
        return y*y_divisions + x
    
    ################## Properties ##################
    @property
    def full_size(self) -> typing.Tuple[Height, Width]:
        sub_h, sub_w = self.sub_size
        return  sub_h*self.y_divisions, sub_w*self.x_divisions

    @property
    def sub_size(self) -> typing.Tuple[Height, Width]:
        return self.subimages[0].window.h, self.subimages[0].window.w

    @property
    def x_divisions(self) -> int:
        return len(self.subimages) // self.y_divisions

    ################## Saving ##################
    def to_image(self) -> Image:
        '''Reconstruct a canvas from a list of subcanvases.'''
        print(f'{self.full_size=}')
        im = np.zeros(self.full_size + (3,), dtype=np.float64)
        for i,si in enumerate(self.subimages):
            im[si.window.y:si.window.y+si.window.h, si.window.x:si.window.x+si.window.w, :] = si.im
        return Image(im=im)

if __name__ == '__main__':
    ig = ImageGrid([])
    print(ig[0,0])





