import os
import pandas as pd
import numpy
import torch
import time
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
from sklearn.model_selection import train_test_split


def mismatch(array):
    max_num = array[0][0]
    i_pos = 0
    j_pos = 0
    array_sum = 0
    for d1 in range(len(array)):
        for d2 in range(len(array)):
            array_sum += array[d1][d2]
            if array[d1][d2] > max_num:
                max_num = array[d1][d2]
                i_pos = d1
                j_pos = d2
    print(f'Of {array_sum:.0f} mismatches, the most common was', i_pos, 'for', j_pos,
          f'and happened {array[i_pos][j_pos]:.0f} times.')


def match(array):
    min_num = float('inf')
    i_pos = 0
    for d1 in range(len(array)):
        sum_d1 = 0
        for d2 in range(len(array)):
            sum_d1 += array[d1][d2]
        if sum_d1 < min_num:
            min_num = sum_d1
            i_pos = d1
    print('The number', i_pos, f'was mismatched only {min_num:.0f} times.')


def mean_time(array, seasons):
    sum_time = 0
    for d1 in range(seasons):
        sum_time += array[d1][3]
    print(f'Mean time for {seasons} epochs was {sum_time / seasons} seconds.')


def testing(loader, array):
    # Testing loop
    correct, total = 0, 0
    with torch.no_grad():

        # Iterate over the validation data and generate predictions
        for i_tmp, data_tmp in enumerate(loader, 0):
            # Get inputs
            inputs_tmp, targets_tmp = data_tmp

            # Generate outputs
            outputs_tmp = mlp(inputs_tmp)

            # Set total and correct
            _, predicted = torch.max(outputs_tmp.data, 1)
            total += targets_tmp.size(0)
            correct += (predicted == targets_tmp).sum().item()
            for j in range(len(predicted)):
                if predicted[j] != targets_tmp[j]:
                    array[targets[j]][predicted[j]] += 1
    return 100.0 * correct / total, array


class MLP(nn.Module):
    """
      Multilayer Perceptron.
    """

    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28 * 1, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 10)
        )

    def forward(self, x):
        """Forward pass"""
        return self.layers(x)


if __name__ == '__main__':
    # Set fixed random number seed
    torch.manual_seed(2727)
    categorization = numpy.zeros((10, 10))

    # Prepare MNIST dataset
    dataset = MNIST(os.getcwd(), train=True, download=True, transform=transforms.ToTensor())
    dataset_validation = MNIST(os.getcwd(), train=True, download=True, transform=transforms.ToTensor())
    dataset_test = MNIST(os.getcwd(), train=False, download=True, transform=transforms.ToTensor())

    _, dataset_validation.data, _, dataset_validation.targets = train_test_split(dataset_validation.data,
                                                                                 dataset_validation.targets,
                                                                                 test_size=1 / 6)

    trainLoader = torch.utils.data.DataLoader(dataset, batch_size=10, shuffle=True, num_workers=1)
    trainLoader_test = torch.utils.data.DataLoader(dataset_test, batch_size=10, shuffle=True, num_workers=1)
    trainLoader_validation = torch.utils.data.DataLoader(dataset_validation, batch_size=10, shuffle=True, num_workers=1)
    # Initialize the MLP
    mlp = MLP()

    # Define the loss function and optimizer
    loss_function = nn.MultiMarginLoss()
    optimizer = torch.optim.Adam(mlp.parameters(), lr=5e-4)

    training_epoch = 100
    array_epoch = []
    array_info = numpy.zeros((training_epoch, 4))

    # Run the training loop for 15 seasons
    for epoch in range(0, training_epoch):

        # Starts timer
        start = time.time()
        # Print epoch
        print(f'Starting epoch {epoch + 1}')

        # Set current loss value
        current_loss = 0.0

        # Iterate over the DataLoader for training data
        for i, data in enumerate(trainLoader, 0):
            # Get inputs`
            inputs, targets = data

            # Zero the gradients
            optimizer.zero_grad()

            # Perform forward pass
            outputs = mlp(inputs)
            # Compute loss
            loss = loss_function(outputs, targets)

            # Perform backward pass
            loss.backward()

            # Perform optimization
            optimizer.step()

            # Print statistics
            current_loss += loss.item()
        end = time.time()
        print(f'Loss after epoch {epoch + 1}:', current_loss / dataset.data.size()[0])
        print('Epoch time needed:', end - start)

        # Print about testing
        print('Starting validating for this epoch')

        # Saving the model
        save_path = './mlp.pth'
        torch.save(mlp.state_dict(), save_path)

        array_info[epoch][0], categorization = testing(trainLoader_validation, categorization)
        print(f'Validation Accuracy: {array_info[epoch][0]}%')

        # Print about testing
        print('Starting testing for this epoch')

        array_info[epoch][1], categorization = testing(trainLoader_test, categorization)

        # Print accuracy
        print(f'Testing Accuracy: {array_info[epoch][1]}%')
        array_epoch.append(f'Epoch {epoch + 1}')
        array_info[epoch][2] = current_loss / dataset.data.size()[0]
        array_info[epoch][3] = end - start
    df = pd.DataFrame(array_info,
                      index=array_epoch, columns=['Validation Accuracy', 'Testing Accuracy', 'Loss', 'Time'])
    df.to_excel('FinalModel.xlsx', sheet_name='sheet_one')

    mean_time(array_info, training_epoch)
    mismatch(categorization)
    match(categorization)
