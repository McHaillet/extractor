import torch.utils.data as data
import pathlib
import mrcfile
import numpy as np
import torch
from tiler import Tiler


class ScoreData(data.Dataset):
    def __init__(self, path: pathlib.Path, patch_size: int = 64, patch_overlap: int = 32):
        """
        Dataset structure for extractor!

        path/
        +- [tomo01]_scores.mrc
        +- [tomo01]_gt.mrc
        +- [tomo02]_scores.mrc
        +- [tomo02]_gt.mrc

        IDs within the brackets can be of any form. The dataset will just try to match files ending in '_scores.mrc'
        and '_gt.mrc'.

        Parameters
        ----------
        path: pathlib.Path
            Input directory with score maps (MRC) and ground truths (MRC) with matching names.
        patch_size: int
            Size of cubic training patches.
        patch_overlap: int
            To what extend the patches should overlap.
        """
        self.path = path
        # viable tomo_ids that have both a _scores and _gt MRC are stored in self.tomo_ids
        tomo_ids = [p.name.replace('_scores.mrc', '') for p in self.path.iterdir() if p.name.endswith('_scores.mrc')]
        self.tomo_ids = [tid for tid in tomo_ids if self.path.joinpath(tid + '_gt.mrc').exists()]
        self.tomo_count = len(self.tomo_ids)

        # load score volumes and label volumes; assume the user provided valid data
        self.score_volumes, self.label_volumes = [], []
        for tid in tomo_ids:
            with mrcfile.open(self.path.joinpath(tid + '_scores.mrc'), mode='r', permissive=True) as mrc:
                self.score_volumes.append(mrc.data.astype(np.float32).copy())
            with mrcfile.open(self.path.joinpath(tid + '_gt.mrc'), mode='r', permissive=True) as mrc:
                self.label_volumes.append(mrc.data.astype(np.float32).copy())

        # split into tiles using patch_size and make a large index of all tiles
        self.tilers = []
        for i in range(self.tomo_count):

            self.tilers.append(Tiler(data_shape=self.score_volumes[i].shape,
                                     tile_shape=(patch_size, patch_size, patch_size),
                                     overlap=(patch_overlap, patch_overlap, patch_overlap),
                                     mode='reflect'))

        self.split = [len(t) for t in self.tilers]

    def __len__(self) -> int:
        """
        Returns
        -------
        int
            The total number of patches in this score dataset.
        """
        return sum(self.split)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        idx: int
            Index of patch that should be returned.

        Returns
        -------
        patch_score, patch_label: tuple[npt.NDArray[float], npt.NDArray[float]]
            Return a patch with corresponding label from the dataset.
        """

        tomo = 0
        for s in self.split:
            idx -= s
            if idx >= 0:
                tomo += 1
                continue
            else:
                idx += s
                break

        # indexing numpy array with [None] adds extra channel dimension
        patch_score = torch.from_numpy(self.tilers[tomo].get_tile(self.score_volumes[tomo], idx)[None])
        patch_label = torch.from_numpy(self.tilers[tomo].get_tile(self.label_volumes[tomo], idx)[None])

        return patch_score, patch_label
