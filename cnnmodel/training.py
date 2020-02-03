import time

import numpy as np
import torch
import tqdm
from torch import optim
from torch.utils.data import DataLoader

from cnnmodel.model import CNNStressNet
from cnnmodel.dataset import CNNDataset


def train(model, device, train_loader, optimizer, epoch, log_interval):
    model.train()
    losses = []
    accuracy = 0
    for batch_idx, ((mfcc, f1_f7), label) in enumerate(tqdm.tqdm(train_loader)):
        mfcc, f1_f7, label = mfcc.to(device), f1_f7.to(device), label.to(device)
        optimizer.zero_grad()
        out = model(mfcc, f1_f7)
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
    test_loss = 0

    accuracy = 0
    with torch.no_grad():
        for batch_idx, ((mfcc, f1_f7), label) in enumerate(tqdm.tqdm(test_loader)):
            mfcc, f1_f7, label = mfcc.to(device), f1_f7.to(device), label.to(device)
            out = model(mfcc, f1_f7)
            test_loss_on = model.loss(out, label).item()
            test_loss += test_loss_on

            pred = torch.argmax(out, dim=1)
            accuracy += torch.sum((pred == label))

            if log_interval is not None and batch_idx % log_interval == 0:
                print('{} Test: [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    time.ctime(time.time()),
                    batch_idx * len(mfcc), len(test_loader.dataset),
                    100. * batch_idx / len(test_loader), test_loss_on))

    test_loss /= len(test_loader.dataset)
    accuracy_mean = (100. * accuracy) / len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} , ({:.4f})%\n'.format(
        test_loss, accuracy, len(test_loader.dataset), accuracy_mean))
    return test_loss, accuracy_mean


def main():
    use_cuda = True
    device = torch.device("cuda" if use_cuda else "cpu")
    print('using device', device)

    import multiprocessing
    print('num cpus:', multiprocessing.cpu_count())

    kwargs = {'num_workers': multiprocessing.cpu_count(),
              'pin_memory': True} if use_cuda else {}

    train_dataset = CNNDataset(root='C:/Users/vivek/dev/capstone/ml-stress-detection-nn/train')
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, **kwargs)

    test_dataset = CNNDataset(root='C:/Users/vivek/dev/capstone/ml-stress-detection-nn/test')
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True, **kwargs)

    model = CNNStressNet(reduction='mean').to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.005)

    for epoch in range(0, 5):
        train_loss, train_accuracy = train(model, device, train_loader, optimizer, epoch, 100)
        test_loss, test_accuracy = test(model, device, test_loader)
        print('After epoch: {}, train_loss: {}, test loss is: {}, train_accuracy: {}, '
              'test_accuracy: {}'.format(epoch, train_loss, test_loss, train_accuracy, test_accuracy))


if __name__ == '__main__':
    main()
