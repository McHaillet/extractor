# torch imports
import torch
import torch.utils.data as data
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel

# other
import logging
import pathlib
import argparse
import os
from extractor.models import UNet3D, PeakFinder
from extractor.data import ScoreData
from extractor.loss import TverskyLoss
from tqdm import tqdm


def setup(
        rank: int,
        world_size: int
):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def prepare(
        dataset: ScoreData,
        rank: int,
        world_size: int,
        batch_size: int,
        pin_memory: bool = False,
        num_workers: int = 0
) -> data.DataLoader:

    sampler = data.distributed.DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        drop_last=True
    )

    return data.DataLoader(
        dataset,
        batch_size=batch_size,
        pin_memory=pin_memory,
        num_workers=num_workers,
        drop_last=True,
        shuffle=False,
        sampler=sampler
    )


def cleanup():
    dist.destroy_process_group()


def train_model(
        rank: int,
        world_size: int,
        train_data_path: pathlib.Path,
        val_fraction: float,
        batch_size: int,
        patch_size: int,
        loss_alpha: float,
        loss_beta: float,
        num_epochs: int,
        output_dir: pathlib.Path,
        log_file: pathlib.Path
):
    if rank == 0:
        logging.basicConfig(filename=log_file, encoding='utf-8', level=logging.DEBUG)

    setup(rank, world_size)

    # Set model to train mode and move to device
    # model = UNet3D(in_channels=1, out_channels=2).to(rank)
    model = PeakFinder().to(rank)
    model = DistributedDataParallel(model, device_ids=[rank])

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    loss_module = TverskyLoss(alpha=loss_alpha, beta=loss_beta)

    dataset = ScoreData(train_data_path, patch_size=patch_size, patch_overlap=patch_size // 2)
    train_dataset, validation_dataset = data.random_split(dataset, [1 - val_fraction, val_fraction])

    train_loader = prepare(train_dataset, rank, world_size, batch_size)
    val_loader = prepare(validation_dataset, rank, world_size, batch_size)

    # Training loop
    for epoch in range(num_epochs):

        train_loader.sampler.set_epoch(epoch)
        val_loader.sampler.set_epoch(epoch)

        training_loss = 0.0

        if rank == 0:
            pbar = tqdm(total=len(train_loader))

        for i, (data_inputs, data_labels) in enumerate(train_loader):

            # set gradients to zero after previous batch
            optimizer.zero_grad()

            # move data to GPU
            data_inputs = data_inputs.to(rank)
            data_labels = data_labels.to(rank)

            # calculate predictions
            preds = model(data_inputs)

            # determine loss
            loss = loss_module(preds, data_labels)
            
            # perform backpropagation
            loss.backward()

            # update parameters
            optimizer.step()

            # take the running average of the loss
            training_loss += loss.item()

            if rank == 0 and i % 10 == 0:
                pbar.update(10)

        if rank == 0:
            pbar.close()

            logging.info(f'[epoch - {epoch + 1}/{num_epochs}; rank - {rank}; training loss - '
                         f' {training_loss / len(train_loader)}]')

        validation_loss = 0.0

        if rank == 0:
            pbar = tqdm(total=len(val_loader))
        # loss should also be evaluated on the validation data so that we can compare training loss and validation loss
        with torch.no_grad():
            for i, (data_inputs, data_labels) in enumerate(val_loader):
                # move data to GPU
                data_inputs = data_inputs.to(rank)
                data_labels = data_labels.to(rank)

                # calculate predictions
                preds = model(data_inputs)

                # determine loss
                loss = loss_module(preds, data_labels)

                validation_loss += loss.item()

                if rank == 0 and i % 10 == 0:
                    pbar.update(10)

        if rank == 0:
            pbar.close()

            logging.info(f'[epoch - {epoch + 1}/{num_epochs}; rank - {rank}; validation loss - '
                         f' {validation_loss / len(val_loader)}]')

        if rank == 0 and epoch % 10 == 0 and epoch > 0:
            torch.save(model.state_dict(), output_dir.joinpath(f'model_epoch-{epoch}.pth'))

    if rank == 0:
        torch.save(model.state_dict(), output_dir.joinpath('model_final.pth'))

    cleanup()


def entry_point():

    parser = argparse.ArgumentParser()
    parser.add_argument('--train-data', type=pathlib.Path, required=True)
    parser.add_argument('--val-fraction', type=float, required=True)
    parser.add_argument('--output-dir', type=pathlib.Path, required=True)
    parser.add_argument('--log-file', type=pathlib.Path, required=True)
    parser.add_argument('--batch-size', type=int, required=False, default=8)
    parser.add_argument('--patch-size', type=int, required=False, default=64)
    parser.add_argument('--loss-alpha', type=float, required=False, default=0.5)
    parser.add_argument('--loss-beta', type=float, required=False, default=0.5)
    parser.add_argument('--epochs', type=int, required=False, default=100)
    parser.add_argument('--gpus', type=int, required=True, default=1,
                        help='number of gpus to train on, assumes gpus have been set via CUDA_VISIBLE_DEVICES')
    args = parser.parse_args()

    mp.spawn(
        train_model,
        args=(
            args.gpus,
            args.train_data,
            args.val_fraction,
            args.batch_size,
            args.patch_size,
            args.loss_alpha,
            args.loss_beta,
            args.epochs,
            args.output_dir,
            args.log_file,
        ),
        nprocs=args.gpus
    )

