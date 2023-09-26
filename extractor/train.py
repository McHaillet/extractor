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
from extractor.models import UNet3D
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
        num_epochs: int,
        output_dir: pathlib.Path
):
    # Set model to train mode and move to device
    model = UNet3D(in_channels=1, out_channels=2)
    model.train()
    model.to(rank)
    model = DistributedDataParallel(model, device_ids=[rank], output_device=rank, find_unused_parameters=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    loss_module = TverskyLoss(classes=2)

    dataset = ScoreData(train_data_path, patch_size=64, patch_overlap=32)
    train_dataset, validation_dataset = data.random_split(dataset, [1 - val_fraction, val_fraction])

    train_loader = prepare(train_dataset, rank, world_size, batch_size)
    val_loader = prepare(validation_dataset, rank, world_size, batch_size)

    # Training loop
    for epoch in range(num_epochs):

        train_loader.sampler.set_epoch(epoch)
        val_loader.sampler.set_epoch(epoch)

        training_loss = 0.0
        pbar = tqdm(total=len(train_loader))
        for i, (data_inputs, data_labels) in enumerate(train_loader):

            # move data to GPU
            data_inputs = data_inputs.to(rank)
            data_labels = data_labels.to(rank)

            # calculate predictions
            preds = model(data_inputs)

            # determine loss
            loss = loss_module(preds, data_labels)

            # set gradients to zero after previous batch
            optimizer.zero_grad()
            # perform backpropagation
            loss.backward()

            # update parameters
            optimizer.step()

            # take the running average of the loss
            training_loss += loss.item()

            if i % 10 == 0:
                pbar.update(10)
        pbar.close()

        logging.info(f'epoch {epoch}: training loss {training_loss / len(train_loader)}')

        validation_loss = 0.0
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

                if i % 10 == 0:
                    pbar.update(10)

        pbar.close()

        logging.info(f'epoch {epoch}: validation loss {validation_loss / len(val_loader)}')

        if epoch % 10 == 0 and epoch > 0:
            torch.save(model.state_dict(), output_dir.joinpath(f'model_epoch-{epoch}.pth'))

    torch.save(model.state_dict(), output_dir.joinpath('model_final.pth'))


def entry_point():

    parser = argparse.ArgumentParser()
    parser.add_argument('--train-data', type=pathlib.Path, required=True)
    parser.add_argument('--val-fraction', type=float, required=True)
    parser.add_argument('--output-dir', type=pathlib.Path, required=True)
    parser.add_argument('--log-file', type=pathlib.Path, required=True)
    parser.add_argument('--batch-size', type=int, required=False, default=8)
    parser.add_argument('--epochs', type=int, required=False, default=100)
    parser.add_argument('--gpus', type=int, required=True, default=1,
                        help='number of gpus to train on, assumes gpus have been set via CUDA_VISIBLE_DEVICES')
    args = parser.parse_args()

    logging.basicConfig(filename=args.log_file, encoding='utf-8', level=logging.DEBUG)

    mp.spawn(
        train_model,
        args=(
            args.gpus,
            args.train_data,
            args.val_fraction,
            args.output_dir,
            args.batch_size,
            args.epochs,
            args.output_dir,
        ),
        nprocs=args.gpus
    )

