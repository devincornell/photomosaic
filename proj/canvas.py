from __future__ import annotations
import pathlib
import dataclasses
import typing
from PIL import Image, ImageChops
import skimage
import numpy as np
import random

from .util import imread_transform, write_as_uint

@dataclasses.dataclass(frozen=True)
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
    
    def new_subcanvasscore(self, dist: float) -> SubCanvasScore:
        return SubCanvasScore(
            dist = dist,
            canvas = self,
        )

@dataclasses.dataclass
class Distances:
    canvas: CanvasBase
    def composit(self, other: CanvasBase) -> float:
        return self.euclid(other) + self.sobel(other)

    def euclid(self, other: CanvasBase) -> float:
        return np.linalg.norm(self.canvas.im - other.im)
            
    def sobel(self, other: CanvasBase) -> float:
        return np.linalg.norm(self.canvas.transform_sobel() - other.transform_sobel())

@dataclasses.dataclass(frozen=True)
class Canvas(CanvasBase):
    fpath: pathlib.Path

    @classmethod
    def read_image(cls, fpath: pathlib.Path) -> Canvas:
        return cls(
            fpath=fpath,
            im=imread_transform(fpath),
        )
    
    def split_subcanvases(self, y_divisions: int, x_divisions: int) -> typing.List[SubCanvas]:
        '''Create by dividing up a single canvas into equal parts.'''
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
    
    
@dataclasses.dataclass(frozen=True)
class SubCanvas(CanvasBase):
    canvas: Canvas

    @classmethod
    def from_canvas(cls, canvas: Canvas, x: int, y: int, w: int, h: int) -> SubCanvas:
        '''Create a subcanvas from a square of the original canvas.'''
        im = canvas.im[y:y+h,x:x+w]
        return cls(
            canvas=canvas,
            im=im,
        )
        

@dataclasses.dataclass(order=True, frozen=True)
class SubCanvasScore:
    dist: float
    canvas: typing.Union[SubCanvas, Canvas]
    
class SubCanvasScores(typing.List[SubCanvasScore]):
    '''Contains a set of subcanvases. Used to reduce two subcanvases with the best solutions.'''
            
    @classmethod
    def from_distances(cls, scores: typing.List[typing.Tuple[int, float, Canvas]]) -> SubCanvasScores:
        '''Place the subcanvas with the best score in the right location'''
        locations = dict()
        finished_sc = set()
        for ind, dist, sc in sorted(scores, key=lambda x: x[1]):
            source_fname = sc.fpath
            if ind not in locations and source_fname not in finished_sc:
                locations[ind] = sc.new_subcanvasscore(dist)
                finished_sc.add(source_fname)
                
        return cls.from_location_dict(locations)

    @classmethod
    def from_location_dict(cls, d: typing.Dict[int, SubCanvasScores]) -> SubCanvasScores:
        '''Make a set of subcanvases from a dictionary where ind corresponds to each subcanvas.'''
        print(d.keys())
        return cls(d[i] for i in range(len(d)))
    
    def to_canvas(self, width: int) -> Canvas:
        '''Reconstruct a canvas from a list of subcanvases.'''
        return Canvas.from_subcanvases([scs.canvas for scs in self], width)
    
    def reduce_subcanvasscores(self, other: SubCanvasScores) -> SubCanvasScores:
        '''Returns a composite container with the best subcanvases in each square.'''
        new_scs = self.__class__()
        for scs, oscs in zip(self, other):
            if scs.dist < oscs.dist:
                new_scs.append(scs)
            else:
                new_scs.append(oscs)
        return new_scs    

