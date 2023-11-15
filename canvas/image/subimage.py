from __future__ import annotations
import dataclasses
import typing
import skimage
import numpy as np
import pathlib

from .image import Image, Height, Width

class PixelX(int):
    pass

class PixelY(int):
    pass

@dataclasses.dataclass
class Window:
    y: PixelY
    x: PixelX
    h: Height
    w: Width

@dataclasses.dataclass(frozen=True)
class SubImage(Image):
    window: Window

    @classmethod
    def from_image(cls, image: Image, y: int, x: int, h: int, w: int) -> SubImage:
        '''Create a subimage from a square of the original canvas.'''
        return cls(
            im = image.im[y:y+h,x:x+w], 
            window = Window(y=y, x=x, h=h, w=w),
        )


