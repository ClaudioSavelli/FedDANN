import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler

def get_train_valid_loader(data_dir,
                           batch_size,
                           augment,
                           random_seed,
                           valid_size=0.1,
                           shuffle=True):

    normalize = transforms.Normalize(
        mean=0.1736,
        std=0.3248,
    )

    # define transforms
    valid_transform = transforms.Compose([
            transforms.ToTensor(), 
            normalize
    ])
    
    train_transform = transforms.Compose([
            transforms.ToTensor(),
            normalize
        ])

    # load the dataset
    train_dataset = datasets.EMNIST(
        root=data_dir, train=True,split = 'byclass',
        download=True, transform=train_transform,
    )

    valid_dataset = datasets.EMNIST(
        root=data_dir, train=True, split = 'byclass',
        download=True, transform=valid_transform,
    )
    
    num_train = len(train_dataset)
    indices = list(range(num_train))
    split = int(np.floor(valid_size * num_train))

    if shuffle:
        np.random.seed(random_seed)
        np.random.shuffle(indices)

    train_idx, valid_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, sampler=train_sampler)
 
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=batch_size, sampler=valid_sampler)

    return (train_loader, valid_loader)


def get_test_loader(data_dir,
                    batch_size,
                    shuffle=True):

    # define transform
    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    dataset = datasets.EMNIST(
        root=data_dir, train=False, split = 'byclass',
        download=True, transform=transform,
    )

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle
    )

    return data_loader

def evaluate_mean_std():
    train_dataset = datasets.EMNIST(
        root='./data', train=True,split = 'byclass',
        download=True
    )

    mean = 0 
    std = 0
    for X, _ in train_dataset: 
        convert_tensor = transforms.ToTensor()

        X = convert_tensor(X)
        mean += X.mean()
        std += X.std()
    mean = mean/len(train_dataset)
    std = std/len(train_dataset)
    print("mean: ",mean," std: ",std)

    return mean, std




# emnist dataset 
#train_loader, valid_loader = get_train_valid_loader(data_dir = './data', batch_size = 64, augment = False, random_seed = 1)

#test_loader = get_test_loader(data_dir = './data', batch_size = 64)
#evaluate_mean_std()