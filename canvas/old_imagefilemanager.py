from __future__ import annotations
import pathlib
import dataclasses
import typing
from PIL import Image
import numpy as np
import random
import skimage
import math
import json

from .canvas import Canvas, SubCanvas

@dataclasses.dataclass
class SourceImage:
    img_path: pathlib.Path
    metadata: typing.Dict[str, typing.Any]
    
    @classmethod
    def from_files(cls, 
        img_path: pathlib.Path,
    ) -> SourceImage:
        with img_to_json_path(img_path).open('r') as f:
            metadata = json.load(f)
        
        return cls(
            img_path = img_path,
            metadata = metadata,
        )
    
    def is_usable(self) -> bool:
        origin = self.metadata.get('googlePhotosOrigin')
        if origin is None:
            return False
        
        try:
            return origin['mobileUpload']['deviceFolder']['localFolderName'] == ''
        except KeyError as e:
            return False
    
    def get_source_md(self) -> typing.Dict[str, typing.Dict]:
        # NOTE: will there be an issue there? Is it irrecoverable? need to check
        return self.metadata.get('googlePhotosOrigin')
    
    @property
    def json_path(self) -> pathlib.Path:
        return img_to_json_path(self.img_path)
    
    @staticmethod
    def check_json_exists(img_path: pathlib.Path) -> bool:
        return img_to_json_path(img_path).exists()

@dataclasses.dataclass
class ImageFileManager:
    root_path: pathlib.Path
    
    @classmethod
    def new(cls, root_path: pathlib.Path) -> ImageFileManager:
        return cls(root_path=pathlib.Path(root_path))
    
    def read_files(self, extensions: typing.Tuple[str] = ('png')) -> typing.List[SourceImage]:
        '''Find all relevent files.'''
        source_images = [p for ext in extensions for p in self.root_path.rglob(f"*.{ext}")]
        print(f'{len(source_images)=}')
        source_images = [SourceImage.from_files(p) for p in source_images if SourceImage.check_json_exists(p)]
        print(f'{len(source_images)=}')
        return source_images
        
    def get_usable_files(self, extensions: typing.Tuple[str] = ('png')) -> typing.List[SourceImage]:
        '''Find all relevent files.'''
        sources = self.read_files(extensions=extensions)
        print(f'{len(sources)=}')
        return [s for s in sources if s.is_usable()]  

def img_to_json_path(img_path: pathlib.Path) -> pathlib.Path:
    return img_path.with_suffix(str(img_path.suffix)+'.json')
