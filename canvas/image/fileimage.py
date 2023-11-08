from __future__ import annotations
import dataclasses
import typing
import skimage
import numpy as np
import pathlib

from .image import Image

@dataclasses.dataclass(frozen=True)
class FileImage(Image):
    path: pathlib.Path

    @classmethod
    def read(cls, path: pathlib.Path) -> FileImage:
        return cls(
            path=pathlib.Path(path),
            im=skimage.io.imread(str(path))
        )
