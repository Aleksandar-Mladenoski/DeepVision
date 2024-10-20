# Training file for on the fly augmentation, my first attempt at doing augmentation


import torch
from torch.utils.data import Dataset, DataLoader, Subset
import numpy as np
import matplotlib.pyplot as plt
from dataset import ImagesDataset
from sklearn.model_selection import train_test_split
from architecture import SimpleCNN
from tqdm import tqdm
from tensorboardX import SummaryWriter
from augmentdata import TransformedImagesDataset, stacking as augment_stacking
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

    dataloader_train = DataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn=augment_stacking)
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
        if counter == 7:
            torch.save(network.state_dict(), "model.pth")
            break
        for trans_imgs_stacked, trans_names, indexes_stacked, class_ids_stacked, class_names, img_paths in (
        tqdm(dataloader_train) if show_progress else dataloader_train):
            trans_imgs_stacked, trans_names, indexes_stacked, class_ids_stacked, class_names, img_paths = trans_imgs_stacked.to(device), trans_names, indexes_stacked.to(device), class_ids_stacked.to(device), class_names, img_paths

            optimizer.zero_grad()
            output = network(trans_imgs_stacked)
            loss = loss_fn(output, class_ids_stacked.squeeze())
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
        correct = 0
        total = 0
        with torch.no_grad():
            for images, classids, classnames, image_filepaths in (
            tqdm(dataloader_eval) if show_progress else dataloader_eval):
                images, classids, classnames, image_filepaths = images.to(device), classids.to(
                    device), classnames, image_filepaths

                output = network(images)
                loss = loss_fn(output, classids.squeeze())

                _, predicted = torch.max(output.data, 1)
                total += classids.squeeze().size(0)
                correct += (predicted == classids.squeeze()).sum().item()

                mbl_eval.append(loss.item())
                minibatch_losses_eval.append(loss.item())
                if writer:
                    writer.add_scalar('Loss/Eval_Minibatch', loss.item(), len(minibatch_losses_eval))

            epoch_losses_eval.append(np.mean(mbl_eval))
            print(f"Accuracy of the network on the test images: {100 * correct // total} %")

        if writer:
            writer.add_scalar('Loss/Eval_Epoch', np.mean(mbl_eval), epoch)

        if old_loss - np.mean(mbl_eval) < 0.05:
            counter += 1
        else:
            counter = 0
        old_loss = np.mean(mbl_eval)
    
    torch.save(network.state_dict(), "model.pth")
    return minibatch_losses_train, minibatch_losses_eval, epoch_losses_train, epoch_losses_eval



if __name__ == '__main__':
    print("Current working directory:", os.getcwd())
    dataset = ImagesDataset(r'training_data', 100, 100)
    TEST_SIZE = 0.1
    BATCH_SIZE = 64
    SEED = 42

    train_indices, test_indices, _, _ = train_test_split(
        range(len(dataset)),
        dataset.targets,
        stratify=dataset.targets,
        test_size=TEST_SIZE,
        random_state=SEED
    )

    train_split = Subset(dataset, train_indices)
    test_split = Subset(dataset, test_indices)

    augment_train_split = TransformedImagesDataset(train_split)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    net = SimpleCNN(
        input_channels=1, 
        hidden_channels=[32, 64, 128, 128, 64], 
        fnn_channels=[128, 64, 32],
        num_classes=20, 
        dropout_rate=0.2,
        kernel_size=[5, 5, 5, 5, 5],
        use_batchnormalization=True
    )     
    net.to(device)

    writer = SummaryWriter()

    minibatch_losses_train, minibatch_losses_eval, epoch_losses_train, epoch_losses_eval = training_loop(
        net, augment_train_split, test_split, 50, BATCH_SIZE, 0.034, show_progress=True, writer=writer
    )

    writer.close()

    plot_losses(minibatch_losses_train, minibatch_losses_eval, epoch_losses_train, epoch_losses_eval)
    torch.save(net.state_dict(), "model.pth")