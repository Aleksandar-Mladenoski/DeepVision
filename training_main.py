# I have lots of training files, some where I use no augmentation, some where I use on the spot augmentation, some where I preload
# This is my main training file, in which I preload saved augmented data.

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from preload_dataset import ImagesDataset
from data import stacking
from architecture import SimpleCNN
from tqdm import tqdm
from tensorboardX import SummaryWriter
import os
from train import plot_losses

def training_loop(
        network: torch.nn.Module,
        train_data: torch.utils.data.Dataset,
        eval_data: torch.utils.data.Dataset,
        num_epochs: int,
        batch_size: int,
        learning_rate: float,
        l1_lambda: float = 0.01,
        l2_lambda: float = 0.01,
        show_progress: bool = False,
        writer: SummaryWriter = None,
        pre_optimizer = None,
        pre_lr_scheduler = None,
) -> tuple[list, list, list, list]:
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    dataloader_train = DataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn=stacking)
    dataloader_eval = DataLoader(eval_data, batch_size=batch_size, shuffle=False, collate_fn=stacking)

    minibatch_losses_train = []
    minibatch_losses_eval = []
    epoch_losses_train = []
    epoch_losses_eval = []
    if pre_optimizer is not None:
        optimizer = pre_optimizer
    else:
        optimizer = torch.optim.Adam(network.parameters(), lr=learning_rate)
    
    if pre_lr_scheduler is not None:
        scheduler = pre_lr_scheduler
    else:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    
    loss_fn = torch.nn.CrossEntropyLoss()
    old_loss = 0
    counter = 0
    best_accuracy = 0

    for epoch in range(num_epochs):
        print("-------------------------------------------------")
        print(f"TRAINING: {epoch}")
        print("-------------------------------------------------")

        network.train()
        mbl_train = []
        if counter == 20:
            break
        for images, classids, classnames, image_filepaths in (
        tqdm(dataloader_train) if show_progress else dataloader_train):
            images, classids, classnames, image_filepaths = images.to(device), classids.to(
                device), classnames, image_filepaths

            optimizer.zero_grad()
            output = network(images)
            loss = loss_fn(output, classids.squeeze())
            
            l1_reg = sum(param.abs().sum() for param in network.parameters())
            l2_reg = sum(param.pow(2).sum() for param in network.parameters())
            loss += l1_lambda * l1_reg + l2_lambda * l2_reg

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
                mbl_eval.append(loss.item())
                minibatch_losses_eval.append(loss.item())

                _, predicted = torch.max(output.data, 1)
                total += classids.squeeze().size(0)
                correct += (predicted == classids.squeeze()).sum().item()

                if writer:
                    writer.add_scalar('Loss/Eval_Minibatch', loss.item(), len(minibatch_losses_eval))

            epoch_losses_eval.append(np.mean(mbl_eval))
            accuracy = 100 * correct // total
            print(f"Accuracy of the network on the test images: {accuracy} %")

        if writer:
            writer.add_scalar('Loss/Eval_Epoch', np.mean(mbl_eval), epoch)

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save(network.state_dict(), 'model.pth')
            torch.save(optimizer.state_dict(), 'optimizer.pth')
            torch.save(scheduler.state_dict(), 'scheduler.pth')

        if old_loss - np.mean(mbl_eval) < 0.006:
            counter += 1
        else:
            counter = 0
        old_loss = np.mean(mbl_eval)
        scheduler.step()

    return minibatch_losses_train, minibatch_losses_eval, epoch_losses_train, epoch_losses_eval

if __name__ == '__main__':
    print("Current working directory:", os.getcwd())
    train_dataset = ImagesDataset(r'project/augment_images', 100, 100)
    test_dataset = ImagesDataset(r'project/validation_images', 100, 100)
    
    TEST_SIZE = 0.1 
    BATCH_SIZE = 64
    SEED = 42
    LEARNING_RATE = 0.014

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    net = SimpleCNN(
        input_channels=1, 
        hidden_channels=[32, 64, 128, 128, 128, 256, 256, 256, 512, 512, 512], 
        fnn_channels=[2048, 512, 128, 64],
        num_classes=20,
        dropout_rate_fnn=0.4,
        dropout_rate_cnn=0.20,
        kernel_size=[5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
        use_batchnormalization=True
    )  
    net.to(device)
    #net.load_state_dict(torch.load('model.pth'))
    #optimizer = torch.optim.Adam(net.parameters(), lr=LEARNING_RATE)
    #optimizer.load_state_dict(torch.load('optimizer.pth'))
    #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    #scheduler.load_state_dict(torch.load('scheduler.pth'))
    writer = SummaryWriter()

    minibatch_losses_train, minibatch_losses_eval, epoch_losses_train, epoch_losses_eval = training_loop(
        net, train_dataset, test_dataset, 20, BATCH_SIZE, LEARNING_RATE, l1_lambda=0, l2_lambda=0, show_progress=True, writer=writer
    )


    writer.close()

    plot_losses(minibatch_losses_train, minibatch_losses_eval, epoch_losses_train, epoch_losses_eval)
