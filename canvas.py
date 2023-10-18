from __future__ import annotations
import pathlib
import dataclasses
import typing
from PIL import Image
import numpy as np
import random

@dataclasses.dataclass
class CanvasBase:
    x: np.ndarray

    def size(self) -> typing.Tuple[int,int]:
        return self.x.shape[:2]
    
    def image(self) -> Image.Image:
        return Image.fromarray(self.x)

    def dist(self, other: CanvasBase) -> float:
        return np.linalg.norm(self.x - other.x)

@dataclasses.dataclass
class Canvas(CanvasBase):
    fpath: pathlib.Path

    @classmethod
    def from_image(cls, fpath: pathlib.Path) -> Canvas:
        x = cls.read_img(fpath)
        return cls(
            fpath=fpath,
            x=x,
        )
    
    @staticmethod
    def read_img(fpath: pathlib.Path) -> np.ndarray:
        return np.array(Image.open(str(fpath)))
    
    def split_subcanvases(self, x_divisions: int, y_divisions: int) -> typing.List[SubCanvas]:
        h,w = self.size()
        x_size = w // x_divisions
        y_size = h // y_divisions
        subcanvases = []
        for iy in range(y_divisions):
            for ix in range(x_divisions):
                subcanvases.append(SubCanvas.from_canvas(self, ix*x_size, iy*y_size, x_size, y_size))
        return subcanvases
    
@dataclasses.dataclass
class SubCanvas(CanvasBase):
    canvas: Canvas

    @classmethod
    def from_canvas(cls, canvas: Canvas, x: int, y: int, w: int, h: int) -> SubCanvas:
        x = canvas.x[y:y+h,x:x+w]
        return cls(
            canvas=canvas,
            x=x,
        )
