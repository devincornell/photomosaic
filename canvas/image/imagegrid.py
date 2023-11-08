from __future__ import annotations
import dataclasses
import typing
import skimage
import numpy as np
import pathlib

if typing.TYPE_CHECKING:
    from .subimage import SubImage

from .image import Image, Height, Width

class X(int):
    pass

class Y(int):
    pass

@dataclasses.dataclass
class ImageGrid:
    images: typing.List[Image]
    y_divisions: int
    image_width: Width
    image_height: Height

    ################## factory method constructors ##################
    @classmethod
    def from_fixed_subimages(cls, image: Image, subimage_height: Height, subimage_width: Width) -> typing.List[Image]:
        '''Create by creating a grid of fixed-size sqares that fill the iamge.'''
        full_h,full_w = image.size
        y_divisions = full_h // subimage_height
        x_divisions = full_w // subimage_width
        subimages = []
        for iy in range(y_divisions):
            for ix in range(x_divisions):
                sc = image.slice(
                    y = iy*subimage_height, 
                    x = ix*subimage_width, 
                    h = subimage_height, 
                    w = subimage_width,
                )
                subimages.append(sc)
        return cls(
            images = subimages, 
            y_divisions = y_divisions,
            image_height = full_h,
            image_width = full_w,
        )
    
    @classmethod
    def from_even_divisions(cls, image: Image, y_divisions: int, x_divisions: int) -> typing.List[Image]:
        '''Create by dividing up a single canvas into equal parts.'''
        h,w = image.size
        x_size = w // x_divisions
        y_size = h // y_divisions
        subimages = []
        for iy in range(y_divisions):
            for ix in range(x_divisions):
                sc = image.slice(iy*y_size, ix*x_size, y_size, x_size)
                subimages.append(sc)
        return cls(
            images = subimages, 
            y_divisions = y_divisions,
            image_height = h,
            image_width = w,
        )

    ################## Dunder ##################
    def __getitem__(self, ind: typing.Union[typing.Tuple[Y,X],int]) -> Image:
        '''Get image at (y,x) location.'''
        y,x = ind
        if y >= self.y_divisions:
            raise IndexError(f'y index {y} is gt or eq to width {self.y_divisions}')
        return self.images[x*self.y_divisions + y]
    
    ################## Properties ##################
    @property
    def full_shape(self) -> int:
        return self.y_divisions * self.x_slices

    @property
    def x_slices(self) -> int:
        return len(self.images) // self.y_slices
    
    ################## Saving ##################
    def to_image(cls, image_grid: ImageGrid, width: int) -> Image:
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


if __name__ == '__main__':
    ig = ImageGrid([])
    print(ig[0,0])





