import torch
import torch.utils.data as data
import logging
import pathlib
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
        device=torch.device('cpu')
):
    # Set model to train mode and move to device
    model.train()
    model.to(device)

    # Training loop
    for epoch in tqdm(range(num_epochs)):
        training_loss = 0.0
        for data_inputs, data_labels in train_loader:

            # move data to GPU
            data_inputs = data_inputs.to(device)
            data_labels = data_labels.to(device)

            # calculate predictions
            preds = model(data_inputs)
            preds = preds.squeeze(dim=1)  # ??? Output is [Batch size, 1], but we want [Batch size]

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

        logging.info(f'epoch {epoch}: training loss {training_loss / len(train_loader)}')

        validation_loss = 0.0
        # loss should also be evaluated on the validation data so that we can compare training loss and validation loss
        with torch.no_grad():
            for data_inputs, data_labels in val_loader:
                # move data to GPU
                data_inputs = data_inputs.to(device)
                data_labels = data_labels.to(device)

                # calculate predictions
                preds = model(data_inputs)
                preds = preds.squeeze(dim=1)  # ??? Output is [Batch size, 1], but we want [Batch size]

                # determine loss
                loss = loss_module(preds, data_labels)

                validation_loss += loss.item()

        logging.info(f'epoch {epoch}: validation loss {validation_loss / len(val_loader)}')


def main():
    logging.basicConfig(filename='example.log', encoding='utf-8', level=logging.DEBUG)

    prediction_classes = 2  # two output channels [background, hit]

    train_dataset = ScoreData(pathlib.Path('training_data'))
    validation_dataset = ScoreData(pathlib.Path('validation_data'))

    train_data_loader = data.DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_data_loader = data.DataLoader(validation_dataset, batch_size=16, shuffle=False)

    model = UNet3D(in_channels=1, out_channels=prediction_classes)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    loss_module = TverskyLoss(classes=prediction_classes)

    train_model(
        model,
        optimizer,
        train_data_loader,
        loss_module,
        val_data_loader,
        num_epochs=100,
        device=torch.device('cuda:0')
    )

