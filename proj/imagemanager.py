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
    
    def read_canvas(self) -> Canvas:
        return Canvas(self.read_image(), self.source_fpath)

    def read_image(self) -> np.ndarray:
        if self.thumb_fpath.exists():
            return skimage.io.imread(str(self.thumb_fpath))
        else:
            return self.write_thumb()
    
    def write_thumb(self) -> np.ndarray:
        im = skimage.io.imread(str(self.source_fpath))

        im = skimage.transform.resize(im, self.scale_res)
        if len(im.shape) < 3:
            im = skimage.color.gray2rgb(im)
        elif len(im.shape) > 3:
            im = skimage.color.rgba2rgb(im)

        skimage.io.imsave(str(self.thumb_fpath), skimage.img_as_ubyte(im))
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

    def __iter__(self) -> typing.Iterator[SourceImage]:
        return iter(self.source_images)
    
    def __len__(self) -> int:
        return len(self.source_images)
    
    def batches(self, batch_size: int) -> typing.Iterator[typing.List[SourceImage]]:
        num_batches = math.ceil(len(self.source_images) / batch_size)
        for i in range(num_batches):
            yield self.source_images[i*batch_size:(i+1)*batch_size]

    def random_canvases(self, k: int) -> typing.List[Canvas]:
        fpaths = random.choices(self.source_images, k=k)
        return [Canvas.read_image(fpath) for fpath in fpaths]
    


