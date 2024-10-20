# File in which I previously used to train, I don't recommend using this one as it doesn't have LR scheduler, or L1 L2 reg
# Or any other fancy things like saving optimizer for later training

import torch
from torch.utils.data import Dataset, DataLoader, Subset
import numpy as np
import matplotlib.pyplot as plt
from dataset import ImagesDataset
from sklearn.model_selection import train_test_split
from architecture import SimpleCNN
from tqdm import tqdm
from tensorboardX import SummaryWriter
#from transformeddataset import TransformedImagesDataset
from data import stacking

import os


def plot_losses(train_losses_batch: list, eval_losses_batch: list, train_losses_epoch: list, eval_losses_epoch: list):
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))

    axs[0, 0].plot(train_losses_batch, color='blue', label='Train loss per minibatch')
    axs[0, 0].set_title('Train loss per minibatch')
    axs[0, 0].set_xlabel('Minibatch')
    axs[0, 0].set_ylabel('Loss')
    axs[0, 0].legend()

    axs[0, 1].plot(eval_losses_batch, color='orange', label='Eval loss per minibatch')
    axs[0, 1].set_title('Eval loss per minibatch')
    axs[0, 1].set_xlabel('Minibatch')
    axs[0, 1].set_ylabel('Loss')
    axs[0, 1].legend()

    axs[1, 0].plot(train_losses_epoch, color='green', label='Train loss per epoch')
    axs[1, 0].set_title('Train loss per epoch')
    axs[1, 0].set_xlabel('Epoch')
    axs[1, 0].set_ylabel('Loss')
    axs[1, 0].legend()

    axs[1, 1].plot(eval_losses_epoch, color='red', label='Eval loss per epoch')
    axs[1, 1].set_title('Eval loss per epoch')
    axs[1, 1].set_xlabel('Epoch')
    axs[1, 1].set_ylabel('Loss')
    axs[1, 1].legend()

    plt.tight_layout()
    plt.savefig('loss_plots.pdf', format='pdf')
    plt.show()


def training_loop(
        network: torch.nn.Module,
        train_data: torch.utils.data.Dataset,
        eval_data: torch.utils.data.Dataset,
        num_epochs: int,
        batch_size: int,
        learning_rate: float,
        show_progress: bool = False,
        writer: SummaryWriter = None
) -> tuple[list, list, list, list]:
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    dataloader_train = DataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn=stacking)
    dataloader_eval = DataLoader(eval_data, batch_size=batch_size, shuffle=False, collate_fn=stacking)

    minibatch_losses_train = []
    minibatch_losses_eval = []
    epoch_losses_train = []
    epoch_losses_eval = []
    optimizer = torch.optim.Adam(network.parameters(), lr=learning_rate)
    loss_fn = torch.nn.CrossEntropyLoss()
    old_loss = 0
    counter = 0

    for epoch in range(num_epochs):
        print("-------------------------------------------------")
        print(f"TRAINING: {epoch}")
        print("-------------------------------------------------")

        network.train()
        mbl_train = []
        if counter == 4:
            break
        for images, classids, classnames, image_filepaths in (
        tqdm(dataloader_train) if show_progress else dataloader_train):
            images, classids, classnames, image_filepaths = images.to(device), classids.to(
                device), classnames, image_filepaths


            optimizer.zero_grad()
            output = network(images)
            loss = loss_fn(output, classids.squeeze())
            loss.backward()
            optimizer.step()
            mbl_train.append(loss.item())
            minibatch_losses_train.append(loss.item())
            if writer:
                writer.add_scalar('Loss/Train_Minibatch', loss.item(), len(minibatch_losses_train))

        epoch_losses_train.append(np.mean(mbl_train))

        if writer:
            writer.add_scalar('Loss/Train_Epoch', np.mean(mbl_train), epoch)

        print("-------------------------------------------------")
        print(f"EVAL: {epoch}")
        print("-------------------------------------------------")

        network.eval()
        mbl_eval = []
        with torch.no_grad():
            for images, classids, classnames, image_filepaths in (
            tqdm(dataloader_eval) if show_progress else dataloader_eval):
                images, classids, classnames, image_filepaths = images.to(device), classids.to(
                    device), classnames, image_filepaths

                output = network(images)
                loss = loss_fn(output, classids.squeeze())
                mbl_eval.append(loss.item())
                minibatch_losses_eval.append(loss.item())
                if writer:
                    writer.add_scalar('Loss/Eval_Minibatch', loss.item(), len(minibatch_losses_eval))

            epoch_losses_eval.append(np.mean(mbl_eval))

        if writer:
            writer.add_scalar('Loss/Eval_Epoch', np.mean(mbl_eval), epoch)

        if np.abs(old_loss - np.mean(mbl_eval)) < 0.1:
            counter += 1
        else:
            counter = 0
        old_loss = np.mean(mbl_eval)

    return minibatch_losses_train, minibatch_losses_eval, epoch_losses_train, epoch_losses_eval



if __name__ == '__main__':
    print("Current working directory:", os.getcwd())
    train_dataset = ImagesDataset(r'project/augment_images', 100, 100)
    test_dataset = ImagesDataset(r'project/validation_images', 100, 100)

    TEST_SIZE = 0.1
    BATCH_SIZE = 64
    SEED = 42

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    net = SimpleCNN(
        input_channels=1, 
        hidden_channels=[32, 64, 128, 128], 
        fnn_channels=[4096, 1024, 512, 128, 64],
        num_classes=20,
        dropout_rate_fnn=0.35,
        kernel_size=[5, 5, 5, 5, 5],
        use_batchnormalization=True
    ) 
    net.to(device)

    writer = SummaryWriter()

    minibatch_losses_train, minibatch_losses_eval, epoch_losses_train, epoch_losses_eval = training_loop(
        net, train_dataset, test_dataset, 20, 64, 0.014, show_progress=True, writer=writer
    )

    writer.close()

    plot_losses(minibatch_losses_train, minibatch_losses_eval, epoch_losses_train, epoch_losses_eval)
    torch.save(net.state_dict(), "model.pth")