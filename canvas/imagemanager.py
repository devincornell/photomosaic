from __future__ import annotations
import pathlib
import dataclasses
import typing
from PIL import Image
import numpy as np
import random
import skimage
import math
import tqdm
import multiprocessing
import os

#from .imagefilemanager import SourceImage, ImageFileManager
#from .canvas import Canvas, SubCanvas
#from .util import imread_transform, imread_transform_resize, write_as_uint
from .sourceimage import SourceImage
from .image import FileImage
from .image import Height, Width

@dataclasses.dataclass
class ImageManager:
    source_images: typing.List[SourceImage]
    thumb_folder: pathlib.Path
    scale_res: typing.Tuple[Height, Width]

    @classmethod
    def from_rglob(cls, 
        source_folder: pathlib.Path, 
        thumb_folder: pathlib.Path, 
        scale_res: typing.Tuple[int,int],
        extensions=('png','PNG', 'jpg', 'JPG'),
    ) -> ImageManager:
        source_images = list()
        for ext in extensions:
            source_images += source_folder.rglob(f"*.{ext}")
        
        source_images = [SourceImage.from_fpaths(fpath, thumb_folder, scale_res) for fpath in source_images]
        return cls(
            source_images=source_images,
            thumb_folder=thumb_folder,
            scale_res=scale_res,
        )
    
    ########## Dunder ##########
    def __len__(self) -> int:
        return len(self.source_images)
        
    ########## reading thumbs ##########
    def read_thumbs_parallel(self, 
        batch_size: int = 10, 
        use_tqdm: bool = False, 
        processes: int = os.cpu_count(), 
        limit: int = None
    ) -> typing.Generator[FileImage]:
        '''Create, save, and return photos in multiple threads and return as generator.'''
        batches = self.batch_source_images(batch_size=batch_size)
        with multiprocessing.Pool(processes=processes) as pool:
            for thumb in self.map_unwrap_thumbs(pool.imap_unordered, batches, use_tqdm, limit):
                yield thumb
                
    def read_thumbs(self, 
        batch_size: int = 10, 
        use_tqdm: bool = False, 
        limit: int = None
    ) -> typing.Generator[FileImage]:
        '''Create, save, and return photos in multiple threads and return as generator.'''
        batches = self.batch_source_images(batch_size=batch_size)
        for thumb in self.map_unwrap_thumbs(map, batches, use_tqdm, limit):
            yield thumb
    
    @classmethod
    def map_unwrap_thumbs(cls, 
        map_func: typing.Callable[[typing.Iterable], FileImage], 
        batches: typing.List[typing.List[FileImage]],
        use_tqdm: bool,
        limit: typing.Optional[int],
    ) -> typing.Generator[FileImage]:
        '''Map and unwrap batch thumbnail reads.'''
        i = 0
        it = map_func(cls.thread_retrieve_thumbs, batches)
        if use_tqdm:
            it = tqdm.tqdm(it, total=len(batches))
        for batch in it:
            for thumb in batch:
                yield thumb
                i += 1
                if limit is not None and i > limit:
                    return

    @staticmethod
    def thread_retrieve_thumbs(sis: typing.List[SourceImage]) -> typing.List[FileImage]:
        '''Read and save a set of thumbs.'''
        return [si.retrieve_thumb() for si in sis]
        
    ########## Filtering ##########
    def filter_usable_photo(self, **kwargs) -> ImageManager:
        '''Filters images, keeping only those SourceImages that are usable photos.'''
        return self.filter(lambda si: si.is_usable_photo(), **kwargs)
        
    def filter(self, func: typing.Callable[[SourceImage], bool], use_tqdm: bool = False) -> ImageManager:
        '''Filters the ImageManager, keeping only those SourceImages for which func returns True.'''
        imgs = tqdm.tqdm(self.source_images) if use_tqdm else self.source_images
        return self.copy(
            source_images=[si for si in imgs if func(si)]
        )
    
    ########## Batching ##########
    def batch_image_managers(self, batch_size: int) -> typing.List[ImageManager]:
        '''Return new managers, each with a subset of the original source images.'''
        batched_imans = list()
        for batch in self.batch_source_images(batch_size=batch_size):
            batched_imans.append(self.copy(source_images=batch))
        return batched_imans
    
    def batch_source_images(self, batch_size: int) -> typing.List[typing.List[SourceImage]]:
        '''Gets batches each of size batch_size, and includes the last batch even if it is smaller than batch_size.'''
        # NOTE: will rewrite with lazy data loading in the future
        num_batches = math.ceil(len(self.source_images) / batch_size)
        batches = list()
        for i in range(num_batches):
            batches.append(self.source_images[i*batch_size:(i+1)*batch_size])
        return batches
    
    def chunk_source_images(self, batch_size: int) -> typing.List[typing.List[SourceImage]]:
        '''Gets chunks each of size batch_size, and discards the last chunk if it is smaller than batch_size.'''
        # NOTE: COPY FROM ABOVE EXCEPT I DON"T USE .CEIL()
        # NOTE: will rewrite with lazy data loading in the future
        num_batches = len(self.source_images) // batch_size
        batches = list()
        for i in range(num_batches):
            batches.append(self.source_images[i*batch_size:(i+1)*batch_size])
        return batches
    
    ############# Copying #############
    def copy(self, **new_attributes) -> ImageManager:
        '''Copies the ImageManager, optionally updating attributes.'''
        return self.__class__(**{**dataclasses.asdict(self), **new_attributes})
    