import torch
import mrcfile
import numpy as np
import numpy.typing as npt
from tiler import Tiler, Merger


@torch.no_grad()
def predict(model: torch.nn.Module, data: npt.NDArray[float], batch_size: int = 10, device=torch.device('cpu')):
    # set model to eval and move to device
    model.eval()
    model.to(device)

    # does this return the patch size correctly?
    patch_size = next(model.parameters()).size()[0]

    tiler = Tiler(
        data_shape=data.shape,
        tile_shape=(patch_size,) * 3,
        patch_overlap=(patch_size // 2,) * 3,
        mode='reflect'
    )

    merger = Merger(tiler, window='hamming')

    for batch_id, batch in tiler(data, batch_size=batch_size):

        # batch to torch Tensor

        # preds back to numpy

        merger.add_batch(batch_id, batch_size, model(batch))

    return merger.merge(unpad=True)
