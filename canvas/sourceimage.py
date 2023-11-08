from __future__ import annotations
import pathlib
import dataclasses
import typing
from PIL import Image
import numpy as np
import random
import skimage
import math

#from .imagefilemanager import SourceImage, ImageFileManager
#from .canvas import Canvas, SubCanvas
#from .sourceimage import SourceImage
from .image import FileImage, Height, Width
#from .util import imread_transform, imread_transform_resize, write_as_uint

@dataclasses.dataclass
class Metadata:
    data: typing.Dict[str, typing.Any]

    @classmethod
    def read_json(cls, json_path: pathlib.Path) -> Metadata:
        pass

    def is_photo(self) -> bool:
        origin = self.data.get('googlePhotosOrigin')
        if origin is None:
            return False
        try:
            return origin['mobileUpload']['deviceFolder']['localFolderName'] == ''
        except KeyError as e:
            return False

@dataclasses.dataclass
class SourceImage:
    image_path: pathlib.Path
    thumb_path: pathlib.Path
    json_path: pathlib.Path
    shape: typing.Tuple[Height, Width]

    @classmethod
    def from_fpaths(cls, 
        image_path: pathlib.Path, 
        thumb_folder: pathlib.Path, 
        shape: typing.Tuple[Height, Width],
    ) -> SourceImage:
        assert image_path.exists()
        return cls(
            image_path = image_path,
            thumb_path = img_to_thumb_path(image_path, thumb_folder, shape=shape),
            json_path = img_to_json_path(image_path),
            shape = shape,
        )

    #################   Metadata   #################
    def is_usable_photo(self) -> bool:
        if not self.has_json():
            return False
        return self.read_metadata().is_photo()
    
    def read_metadata(self) -> Metadata:
        return Metadata.read_json(self.json_path)

    def has_json(self) -> bool:
        return self.json_path.exists()

    #################   Image   #################
    def retrieve_thumb(self) -> FileImage:
        '''Read from thumb if exists or make thumb and return it.'''
        if self.has_thumb():
            return self.read_image(self.thumb_path)
        else:
            thumb_image = self.read_image(self.image_path).resize(self.shape)
            thumb_image.as_ubyte().write(self.thumb_path)
            return thumb_image
    
    def has_thumb(self) -> bool:
        return self.thumb_path.exists()
    
    @staticmethod
    def read_image(path: pathlib.Path) -> FileImage:
        '''Read image, convert to float, and transform to RGB.'''
        FileImage.read(path).as_float().transform_color_rgb()
            
def img_to_json_path(img_path: pathlib.Path) -> pathlib.Path:
    return img_path.with_suffix(str(img_path.suffix)+'.json')

def img_to_thumb_path(img_path: pathlib.Path, thumb_folder: pathlib.Path, shape: typing.Tuple[Height, Width]) -> pathlib.Path:
    h,w = shape
    return thumb_folder / f'{img_path.stem}_{h}x{w}.{img_path.suffix[1:]}'
