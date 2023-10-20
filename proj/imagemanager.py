from __future__ import annotations
import pathlib
import dataclasses
import typing
from PIL import Image
import numpy as np
import random
import skimage
import math

from .canvas import Canvas, SubCanvas
from .util import imread_transform, imread_transform_resize, write_as_uint

@dataclasses.dataclass
class SourceImage:
    source_fpath: pathlib.Path
    thumb_fpath: pathlib.Path
    scale_res: typing.Tuple[int,int]

    @classmethod
    def from_fpaths(cls, source_fpath: pathlib.Path, thumb_folder: pathlib.Path, scale_res: typing.Tuple[int,int]) -> SourceImage:
        return cls(
            source_fpath=source_fpath,
            thumb_fpath=cls.get_thumb_path(source_fpath, thumb_folder, scale_res),
            scale_res=scale_res,
        )
    
    @classmethod
    def from_manager(cls, source_fpath: pathlib.Path, manager: ImageManager) -> SourceImage:
        return cls(
            source_fpath=source_fpath,
            thumb_folder=cls.get_thumb_path(source_fpath, manager.thumb_folder, manager.scale_res),
            scale_res=manager.scale_res,
        )

    @staticmethod
    def get_thumb_path(fpath: pathlib.Path, thumb_folder: pathlib.Path, scale_res: typing.Tuple[int,int]) -> pathlib.Path:
        return thumb_folder / f'{fpath.stem}_{scale_res[0]}x{scale_res[1]}.{fpath.suffix[1:]}'
    
    def retrieve_canvas(self) -> Canvas:
        return Canvas(self.retrieve_thumb(), self.source_fpath)

    def retrieve_thumb(self) -> np.ndarray:
        '''Read from thumb if exists or make thumb and return it.'''
        if self.thumb_fpath.exists():
            return imread_transform(self.thumb_fpath)
        else:
            return self.write_thumb()
    
    def write_thumb(self) -> np.ndarray:
        '''Reads original file and writes thumbnail, returning the thumb version.'''
        im = imread_transform_resize(self.source_fpath, self.scale_res)
        write_as_uint(im, self.thumb_fpath)
        return im

@dataclasses.dataclass
class ImageManager:
    source_images: typing.List[SourceImage]
    thumb_folder: pathlib.Path
    scale_res: typing.Tuple[int,int]

    @classmethod
    def from_folders(cls, 
        source_folder: pathlib.Path, 
        thumb_folder: typing.List[pathlib.Path], 
        scale_res: typing.Tuple[int,int],
        extensions=('png',),
    ) -> ImageManager:
        source_images = list()
        for ext in extensions:
            source_images += source_folder.rglob(f"*.{ext}")
        
        source_images = [SourceImage.from_fpaths(fpath, thumb_folder, scale_res) for fpath in source_images]
        random.shuffle(source_images)
        return cls(
            source_images=source_images,
            thumb_folder=thumb_folder,
            scale_res=scale_res,
        )

    #def __iter__(self) -> typing.Iterator[SourceImage]:
    #    return iter(self.source_images)
    
    def __len__(self) -> int:
        return len(self.source_images)


    
    def sample_source_images(self, k: int) -> typing.List[SourceImage]:
        return random.choices(self.source_images, k=k)
    
    ################## DEPRICATED FOR NOW #################
    
    def random_canvases(self, k: int) -> typing.List[Canvas]:
        fpaths = random.choices(self.source_images, k=k)
        return [Canvas.read_image(fpath) for fpath in fpaths]

    def batches(self, batch_size: int) -> typing.Iterator[typing.List[SourceImage]]:
        # NOTE: will rewrite with lazy data loading in the future
        num_batches = math.ceil(len(self.source_images) / batch_size)
        for i in range(num_batches):
            yield self.source_images[i*batch_size:(i+1)*batch_size]
            
            
            