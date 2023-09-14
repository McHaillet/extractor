import torch.utils.data as data
import pathlib
import mrcfile
import tiler


class ScoreData(data.Dataset):
    def __init__(self, path: pathlib.Path, patch_size: int):
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
            input directory with score maps (MRC) and ground truths (MRC) with matching names
        patch_size: int
            size of cubic training patches
        """
        self.path = path
        tomo_ids = [p.name.strip('_scores.mrc') for p in self.path.iterdir() if p.name.endswith('_scores.mrc')]
        # viable tomo_ids that have both a _scores and _gt MRC are stored in self.tomo_ids
        self.tomo_ids = [tid for tid in tomo_ids if self.path.joinpath(tid + '_gt.mrc').exists()]
        self.tomo_count = len(self.tomo_ids)

        # load all data

        # split into tiles using patch_size and make a large index of all tiles

    def __len__(self):
        return self.tile_count

    def __getitem__(self, idx):
        pass

