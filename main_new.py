from __future__ import annotations
#import pathlib
from pathlib import Path
import skimage
import dataclasses
import typing
from PIL import Image
import numpy as np
import random
import tqdm
import multiprocessing
random.seed(0)

#import coproc
import canvas
import mediatools

def greedy_optimize_thread(
    thread_index: int, 
    cand_files: list[mediatools.ImageFile], 
    sgrid: SubimageGrid, 
    outfolder: Path
) -> GridScores:
    best_scores = GridScores.from_grid_size(x_slices=sgrid.x_slices, y_slices=sgrid.y_slices)
    for i, cand_file in tqdm.tqdm(enumerate(cand_files), ncols=100, total=len(cand_files)):

        # transform image to comparable format
        cand_img = cand_file.read().as_float().to_rgb().transform.resize(sgrid.subimage_size)

        # compute distances between candidate and all subimages
        dists = sgrid.calc_distances(cand_img)
        
        # insert the candidate image if it is the best for any subimage
        best_scores.insert_any_if_best(cand_img, dists)

        # save this every iteration so we can see it happen
        if i % 5 == 0:
            best_score_grid = best_scores.get_subimage_grid()
            best_score_grid.recombine().as_ubyte().write(outfolder.joinpath(f'current_{thread_index}.png'))
    return best_scores


def find_best_chunk_thread(args) -> canvas.SubCanvasScores:
    thread_index: int = args[0]
    width: int = args[1]
    source_images: typing.List[canvas.SourceImage] = args[2]
    target_subcanvases: typing.List[canvas.SubCanvas] = args[3]
    outfolder: Path = args[4]
    
    # compute distances for every source image to every target subcanvas
    distances: typing.List[typing.Tuple[int,float,canvas.SubCanvas]] = list()
    for si in tqdm.tqdm(source_images):
        try:
            si_canvas = si.retrieve_canvas()
            for ind, target_sc in enumerate(target_subcanvases):
                d = target_sc.dist.composit(si_canvas)
                distances.append((ind, d, si_canvas))
        except ValueError as e:
            print(f'\ncouldn\'t load image {si.source_fpath}')
        except OSError as e:
            print(f'\ncouldn\'t load image {si.source_fpath}')
    scs = canvas.SubCanvasScores.from_distances(distances)
    best = scs.to_canvas(width)
    best.write_image(outfolder.joinpath(f'current_{thread_index}.png'))
    print(f'\nfinished {thread_index}')
    return scs

GridIndex = int

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

    def get_subimage_grid(self) -> SubimageGrid:
        '''Get the best images as a subimage grid.'''
        subimages = [s.image for s in self.scores]
        return SubimageGrid(
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


@dataclasses.dataclass
class SubimageGrid:
    '''A grid of subimages.'''
    subimages: list[mediatools.Image|None]
    x_slices: int
    y_slices: int
    
    @classmethod
    def from_image(cls, im: mediatools.Image, x_slices: int, y_slices: int) -> typing.Self:
        '''From an image, create a grid of subimages.'''
        h, w = im.size
        y_slice_size, x_slice_size = h // y_slices, w // x_slices
        
        subimages: list[mediatools.Image] = list()
        for y in range(y_slices):
            for x in range(x_slices):
                subimages.append(im[y*y_slice_size:(y+1)*y_slice_size, x*x_slice_size:(x+1)*x_slice_size])
        return cls(
            subimages=subimages,
            x_slices=x_slices,
            y_slices=y_slices,
        )
    
    def recombine(self) -> mediatools.Image:
        '''Recombine the subimages into a single image.'''
        h, w = self.subimage_size
        im = np.zeros((h*self.y_slices, w*self.x_slices, 3), dtype=np.float64)
        for i, subimage in enumerate(self.subimages):
            if subimage is not None:
                y, x = i // self.x_slices, i % self.x_slices
                im[y*h:(y+1)*h, x*w:(x+1)*w] = subimage.im
        return mediatools.Image(im)

    @property
    def subimage_size(self):
        for si in self.subimages:
            if si is not None:
                return si.size
        raise ValueError('No subimages found in grid.')

    def calc_distances(self, cand_img: mediatools.Image) -> list[tuple[float, GridIndex]]:
        '''Calculate distances between each subimage and a candidate image.'''
        return [(si.dist.composit(cand_img), i) for i, si in enumerate(self.subimages)]

    def __getitem__(self, i: GridIndex | slice) -> mediatools.Image | list[mediatools.Image]:
        '''Slice into subimage list.'''
        return self.subimages[i]
    
    def __len__(self):
        return len(self.subimages)
    
    def __iter__(self):
        return iter(self.subimages)




def main():
    
    output_fname = 'data/tmp/best_score_grid.png'

    base_file = mediatools.ImageFile.from_path("data/targets/black_circle1.png")
    base_image = base_file.read().transform.resize((500, 500))
    sgrid = SubimageGrid.from_image(base_image, 10, 10)
    print(base_image)

    cand_files = mediatools.ImageFiles.from_rglob('data/dataset_coco/train/')
    print(len(cand_files))

    best_scores = greedy_optimize_thread(0, cand_files[:None], sgrid, Path('data/tmp/'))
    
    best_score_grid = best_scores.get_subimage_grid()
    best_score_grid.recombine().as_ubyte().write(output_fname)

    return
    best_scores = GridScores.from_grid_size(x_slices=x_slices, y_slices=y_slices)
    for cand_file in tqdm.tqdm(cand_files, ncols=100):
        cand_img = cand_file.read().as_float().to_rgb().transform.resize(sgrid.subimage_size)
        dists = sgrid.calc_distances(cand_img)
        
        best_scores.insert_any_if_best(cand_img, dists)

        # run this every time so we can see it in action
        #try:
        best_score_grid = best_scores.get_subimage_grid()
        best_score_grid.recombine().as_ubyte().write(output_fname)
        #except AttributeError as e:
            #print('index error')
            #continue
        #except KeyError as e:
            #print('key error')
        #    continue

    return
    print(len(base_grid))
    for i, im in enumerate(base_grid):
        im.as_ubyte().write(Path(f'data/tmp/split_{i}.png'))


    return

    target = canvas.Canvas.read_image(
        #fpath=Path("data/targets/Gray-Mountain_small.webp"),
        fpath=Path("data/targets/lofi_tiny.jpg"),
    )
    
    outfolder = Path("data/output/lofi-30x30/")
    outfolder.mkdir(exist_ok=True, parents=True)

    the_monitor = coproc.Monitor(
        fig_path=outfolder.joinpath('progress.png'), 
        log_path=outfolder.joinpath('progress.log'), 
        snapshot_seconds=1, 
        save_fig_freq=1
    )

    with the_monitor as monitor:
        monitor.add_note(target.im.dtype, target.im.shape, do_print=True)    
            
        height, width = 30, 30
        subtargets = list(target.split_subcanvases(height, width))
        
        imman = canvas.ImageManager.from_file_manager(
            manager=canvas.ImageFileManager(
                root_path = Path('/StorageDrive/unzipped_photos/Takeout/'),
            ),
            thumb_folder=Path("data/personal_thumbs/"),
            scale_res=subtargets[0].size,
            extensions=('png','jpg', 'JPG'),
        )
        monitor.add_note(f'{len(imman)=}', do_print=True)
        #exit()
        batches = [(i,width,bi,subtargets, outfolder) for i,bi in enumerate(imman.chunk_source_images(height * width * 2))]
        
        monitor.add_note(f'running {len(batches)} batches against {len(subtargets)} subcanvases')
        with multiprocessing.Pool(12) as pool:
            #monitor.update_child_processes()
            
            map_func = pool.imap_unordered
            scss: typing.List[canvas.SubCanvasScores] = list()
            for i, r in enumerate(map_func(find_best_chunk_thread, batches)):
                scss.append(r)
                monitor.add_note(f'finished batch {i}')
                
        monitor.add_note('finished all batches')
        best = scss[0]
        for i in range(1,len(scss)):
            best = best.reduce_subcanvasscores(scss[i])
        best.to_canvas(width).write_image(outfolder.joinpath(f'final.png'))
    

if __name__ == "__main__":
    main()




