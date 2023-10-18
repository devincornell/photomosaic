from __future__ import annotations
import pathlib
import dataclasses
import typing
from PIL import Image
import numpy as np
import random

@dataclasses.dataclass
class ImageManager:
    target_fpath: pathlib.Path
    source_fpaths: typing.List[pathlib.Path]

    @classmethod
    def from_folders(cls, target_fpath: pathlib.Path, source_folder: typing.List[pathlib.Path], extensions=('png',)) -> ImageManager:
        source_fpaths = list()
        for ext in extensions:
            source_fpaths += source_folder.rglob(f"*.{ext}")
        
        return cls(
            target_fpath=target_fpath,
            source_fpaths=source_fpaths,
        )

