from __future__ import annotations
import dataclasses
import typing
import skimage
import numpy as np
import pathlib

if typing.TYPE_CHECKING:
    from .image import Image


class X(int):
    pass

class Y(int):
    pass

@dataclasses.dataclass
class ImageGrid:
    images: typing.List[Image]
    width: int

    def __getitem__(self, ind: typing.Union[typing.Tuple[X,Y],int]) -> Image:
        '''Get image at index or (y,x) index.'''
        if isinstance(ind, int):
            return self.images[ind]
        else:
            x,y = ind
            if x >= self.width:
                raise IndexError(f'x index {x} is gt or eq to width {self.width}')
            return self.images[y*self.width + x]



if __name__ == '__main__':
    ig = ImageGrid([])
    print(ig[0,0])





