import torch
import torch.utils.data as data
import logging
import pathlib
import argparse
from extractor.models import UNet3D
from extractor.data import ScoreData
from extractor.loss import TverskyLoss
from tqdm import tqdm


def train_model(
        model,
        optimizer,
        loss_module,
        train_loader,
        val_loader,
        num_epochs=100,
        device=torch.device('cpu'),
        output_dir=None
):
    # Set model to train mode and move to device
    model.train()
    model.to(device)

    # Training loop
    for epoch in range(num_epochs):
        training_loss = 0.0
        pbar = tqdm(total=len(train_loader))
        for i, (data_inputs, data_labels) in enumerate(train_loader):

            # move data to GPU
            data_inputs = data_inputs.to(device)
            data_labels = data_labels.to(device)

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
                data_inputs = data_inputs.to(device)
                data_labels = data_labels.to(device)

                # calculate predictions
                preds = model(data_inputs)

                # determine loss
                loss = loss_module(preds, data_labels)

                validation_loss += loss.item()

                if i % 10 == 0:
                    pbar.update(10)

        pbar.close()

        logging.info(f'epoch {epoch}: validation loss {validation_loss / len(val_loader)}')

        if output_dir is not None and epoch % 10 == 0 and epoch > 0:
            torch.save(model.state_dict(), output_dir.joinpath(f'model_epoch-{epoch}.pth'))


def entry_point():

    parser = argparse.ArgumentParser()
    parser.add_argument('--train-data', type=pathlib.Path, required=True)
    parser.add_argument('--val-data', type=pathlib.Path, required=True)
    parser.add_argument('--output-dir', type=pathlib.Path, required=True)
    parser.add_argument('--log-file', type=pathlib.Path, required=True)
    parser.add_argument('--batch-size', type=int, required=False, default=8)
    parser.add_argument('--epochs', type=int, required=False, default=100)
    parser.add_argument('--gpu-id', type=int, required=False)
    args = parser.parse_args()

    logging.basicConfig(filename=args.log_file, encoding='utf-8', level=logging.DEBUG)

    train_dataset = ScoreData(args.train_data)
    validation_dataset = ScoreData(args.val_data)

    train_data_loader = data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_data_loader = data.DataLoader(validation_dataset, batch_size=args.batch_size, shuffle=False)

    model = UNet3D(in_channels=1, out_channels=2, dropout=0.2)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    loss_module = TverskyLoss(classes=2)

    train_model(
        model,
        optimizer,
        loss_module,
        train_data_loader,
        val_data_loader,
        num_epochs=args.epochs,
        device=torch.device(f'cuda:{args.gpu_id}') if args.gpu_id is not None else torch.device('cpu'),
        output_dir=args.output_dir
    )

    torch.save(model.state_dict(), args.output_dir.joinpath('model_final.pth'))

