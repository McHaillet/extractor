import torch
import torch.nn.functional as F
import mrcfile
import numpy.typing as npt
import argparse
import pathlib
import numpy as np
from tqdm import tqdm
from extractor.models import UNet3D
from tiler import Tiler, Merger


@torch.no_grad()
def predict(model: torch.nn.Module, data: npt.NDArray[float], batch_size: int = 10, device=torch.device('cpu')):
    # set model to eval and move to device
    model.eval()
    model.to(device)

    # tomo = torch.from_numpy(data).unsqueeze(dim=0).unsqueeze(dim=0)
    # print(tomo.shape)
    # preds = model(tomo)[0]
    # print(preds.shape)

    # does this return the patch size correctly?
    patch_size = next(model.parameters()).size()[0]

    tiler = Tiler(
        data_shape=data.shape,
        tile_shape=(patch_size,) * 3,
        overlap=(patch_size // 2,) * 3,
        mode='reflect'
    )

    merger = Merger(tiler, window='hamming')

    pbar = tqdm(total=tiler.n_tiles)

    for tile_id, tile in tiler(data):

        batch = torch.from_numpy(tile).unsqueeze(dim=0).unsqueeze(dim=0).to(device)
        prob = F.softmax(model(batch), dim=1)
        merger.add(tile_id, prob[0, 1].to(torch.device('cpu')).numpy())

        pbar.update(1)
    pbar.close()

    return merger.merge(unpad=True)


def entry_point():
    parser = argparse.ArgumentParser()
    parser.add_argument('--score-map', type=pathlib.Path, required=True)
    parser.add_argument('--output', type=pathlib.Path, required=True)
    parser.add_argument('--model', type=pathlib.Path, required=True)
    parser.add_argument('--confidence', type=float, required=False, default=True,
                        help='How confident the predictions should be, expressed as probability. A value close to 1 '
                             'allows only very confident predictions, values closer to 0 will select more points at '
                             'the expense of increasing false positives. The default is 0.5 ')
    parser.add_argument('--voxel-size', type=float, required=True)
    parser.add_argument('--gpu-id', type=int, required=False)
    args = parser.parse_args()
    data = mrcfile.read(args.score_map)  # also read voxel_size
    model = UNet3D(in_channels=1, out_channels=2, dropout=0.2)
    model.load_state_dict(torch.load(args.model))
    result = predict(
        model, 
        data, 
        batch_size=2, 
        device=torch.device(f'cuda:{args.gpu_id}') if args.gpu_id is not None else torch.device('cpu')
    )
    binary_result = np.zeros_like(result)
    binary_result[result > args.confidence] = 1
    mrcfile.write(args.output, binary_result, voxel_size=args.voxel_size, overwrite=True)
