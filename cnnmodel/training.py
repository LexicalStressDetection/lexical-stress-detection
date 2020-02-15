import sys
import time
import tqdm

import numpy as np
import torch

from torch import optim
from torch.utils.data import DataLoader

from cnnmodel.model import CNNStressNet
from cnnmodel.dataset import CNNDataset

from util.pt_util import restore_objects, save_model, save_objects, restore_model


def train(model, device, train_loader, optimizer, epoch, log_interval):
    model.train()
    losses = []
    accuracy = 0
    for batch_idx, ((mfcc, non_mfcc), label) in enumerate(tqdm.tqdm(train_loader)):
        mfcc, non_mfcc, label = mfcc.to(device), non_mfcc.to(device), label.to(device)
        optimizer.zero_grad()
        out = model(mfcc, non_mfcc)
        loss = model.loss(out, label)

        with torch.no_grad():
            pred = torch.argmax(out, dim=1)
            accuracy += torch.sum((pred == label))

        losses.append(loss.item())
        loss.backward()
        optimizer.step()

        if batch_idx % log_interval == 0:
            print('{} Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                time.ctime(time.time()),
                epoch, batch_idx * len(mfcc), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

    accuracy_mean = (100. * accuracy) / len(train_loader.dataset)

    return np.mean(losses), accuracy_mean


def test(model, device, test_loader, log_interval=None):
    model.eval()
    losses = []

    accuracy = 0
    with torch.no_grad():
        for batch_idx, ((mfcc, non_mfcc), label) in enumerate(tqdm.tqdm(test_loader)):
            mfcc, non_mfcc, label = mfcc.to(device), non_mfcc.to(device), label.to(device)
            out = model(mfcc, non_mfcc)
            test_loss_on = model.loss(out, label).item()
            losses.append(test_loss_on)

            pred = torch.argmax(out, dim=1)
            accuracy += torch.sum((pred == label))

            if log_interval is not None and batch_idx % log_interval == 0:
                print('{} Test: [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    time.ctime(time.time()),
                    batch_idx * len(mfcc), len(test_loader.dataset),
                    100. * batch_idx / len(test_loader), test_loss_on))

    test_loss = np.mean(losses)
    accuracy_mean = (100. * accuracy) / len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} , ({:.4f})%\n'.format(
        test_loss, accuracy, len(test_loader.dataset), accuracy_mean))
    return test_loss, accuracy_mean


def main(train_path, test_path, model_path):
    print('train path: {}'.format(train_path))
    print('test path: {}'.format(test_path))
    print('model path: {}'.format(model_path))

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print('using device', device)

    import multiprocessing
    print('num cpus:', multiprocessing.cpu_count())

    kwargs = {'num_workers': multiprocessing.cpu_count(),
              'pin_memory': True} if use_cuda else {}

    train_dataset = CNNDataset(root=train_path)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, **kwargs)

    test_dataset = CNNDataset(root=test_path)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True, **kwargs)

    model = CNNStressNet(reduction='mean').to(device)
    model = restore_model(model, model_path)
    last_epoch, max_accuracy, train_losses, test_losses, train_accuracies, test_accuracies = \
        restore_objects(model_path, (0, 0, [], [], [], []))

    start = last_epoch + 1 if max_accuracy > 0 else 0

    optimizer = optim.Adam(model.parameters(), lr=0.005)

    for epoch in range(start, 5):
        train_loss, train_accuracy = train(model, device, train_loader, optimizer, epoch, 100)
        test_loss, test_accuracy = test(model, device, test_loader)
        print('After epoch: {}, train_loss: {}, test loss is: {}, train_accuracy: {}, '
              'test_accuracy: {}'.format(epoch, train_loss, test_loss, train_accuracy, test_accuracy))

        train_losses.append(train_loss)
        test_losses.append(test_loss)
        train_accuracies.append(train_accuracy)
        test_accuracies.append(test_accuracy)
        if test_accuracy > max_accuracy:
            max_accuracy = test_accuracy
            save_model(model, epoch, model_path)
            save_objects((epoch, max_accuracy, train_losses, test_losses, train_accuracies, test_accuracies),
                         epoch, model_path)
            print('saved epoch: {} as checkpoint'.format(epoch))


if __name__ == '__main__':
    # needs three command line arguments
    # 1. root path of train data
    # 2. root path of test data
    # 3. path where saved models are saved
    main(sys.argv[1], sys.argv[2], sys.argv[3])
