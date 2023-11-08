from __future__ import annotations
import dataclasses
import typing
import skimage
import numpy as np
import pathlib

from .image import Image

@dataclasses.dataclass(frozen=True)
class SubImage(Image):
    x: int
    y: int
    h: int
    w: int

    @classmethod
    def from_image(cls, image: Image, y: int, x: int, h: int, w: int) -> Image:
        '''Create a subimage from a square of the original canvas.'''
        return cls(
            im = image.im[y:y+h,x:x+w],
            y=y,
            x=x,
            h=h,
            w=w,
        )


