
import typing
import numpy as np
import pathlib
import skimage

def write_as_uint(im: np.ndarray, fpath: pathlib.Path) -> None:
    '''Writes image as float.'''
    skimage.io.imsave(str(fpath), skimage.img_as_uint(im))

def imread_transform_resize(fpath: pathlib.Path, resize_res: typing.Tuple[int,int]) -> np.ndarray:
    '''Read and transform image then scale it according to new resolution'''
    im = imread_transform(fpath)
    return skimage.transform.resize(im, resize_res)

def imread_transform(fpath: pathlib.Path) -> np.ndarray:
    '''Read image while converting colors and changing uint8 to float.'''
    im = skimage.io.imread(str(fpath))
    im = transform_color_rgb(im)
    return skimage.util.img_as_float(im)

def transform_color_rgb(im: np.ndarray) -> np.ndarray:
    if len(im.shape) < 3:
        im = skimage.color.gray2rgb(im)
    elif im.shape[2] > 3:
        im = skimage.color.rgba2rgb(im)
    return im

