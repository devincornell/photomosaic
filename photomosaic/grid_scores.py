from __future__ import annotations
from pathlib import Path
import dataclasses
import typing

import mediatools

from .image_grid import ImageGrid, GridIndex

@dataclasses.dataclass
class ImageScore:
    dist: float
    image: mediatools.Image

    @classmethod
    def empty(cls) -> typing.Self:
        '''Create an empty ImageScore.'''
        return cls(dist=float('inf'), image=None)

@dataclasses.dataclass
class GridScores:
    '''Maintains best candidate edges for each grid position.'''
    # map from grid index to an image/score (should be a list, but will need to know size beforehand).
    scores: list[ImageScore]
    x_slices: int
    y_slices: int

    @classmethod
    def from_grid_size(cls, x_slices: int, y_slices: int) -> typing.Self:
        '''Create a GridScores object from grid size.'''
        return cls(
            scores=[ImageScore.empty() for _ in range(x_slices * y_slices)],
            x_slices=x_slices,
            y_slices=y_slices,
        )

    def get_subimage_grid(self) -> ImageGrid:
        '''Get the best images as a subimage grid.'''
        subimages = [s.image for s in self.scores]
        return ImageGrid(
            subimages=subimages, 
            x_slices=self.x_slices, 
            y_slices=self.y_slices
        )
    
    def insert_any_if_best(self, cand_img: mediatools.Image, dists: list[tuple[float,GridIndex]]) -> None:
        '''Check if an image is the best for a given index.'''
        for dist, grid_index in sorted(dists):
            if self.insert_if_best(cand_img, grid_index, dist):
                break

    def reduce(self, other: typing.Self) -> typing.Self:
        '''Reduce two GridScores into one with the best scores.'''
        
        # make sure everything is good
        assert(self.x_slices == other.x_slices)
        assert(self.y_slices == other.y_slices)
        
        new_scores = self.copy()
        for i, img_score in enumerate(other.scores):
            new_scores.insert_if_best(i, img_score.image, img_score.dist)
        return new_scores

    def insert_if_best(self, cand_img: mediatools.Image, grid_index: GridIndex, dist: float) -> bool:
        '''Check if an image is the best for a given index.'''
        if self.scores[grid_index] is None or dist < self.scores[grid_index].dist:
            self.scores[grid_index] = ImageScore(dist=dist, image=cand_img)
            return True
        return False

    def copy(self) -> typing.Self:
        '''Copy the object.'''
        return self.__class__(
            scores=self.scores.copy(),
            x_slices=self.x_slices,
            y_slices=self.y_slices,
        )
