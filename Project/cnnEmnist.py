import torch 
import torch.nn as nn 
import torch.optim as optim 
from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
from models.cnn1 import My_CNN

import sys

import numpy as np
from math import sqrt

import wandb

lrng_rate = 0.01
training_epochs = 20
p = 0.25
wd = 1e-4

# start a new wandb run to track this script
wandb.init(
    mode="disabled",
    # set the wandb project where this run will be logged
    project="RealEmnistBenchmark",
    
    # track hyperparameters and run metadata
    config={
    "learning_rate": lrng_rate,
    "architecture": "CNN",
    "dataset": "EMNIST",
    "epochs": training_epochs,
    "Optimiser": "SGD(params=model.parameters(), lr=lrng_rate, momentum = 0.9)",
    "criterion": "nn.CrossEntropyLoss()",
    "p": p,
    #"lr modifier": "Multiplied by 0.1 every 5 iterations",
    "seed": 42,
    "weight decay": wd
    }
)

imageDim = 28*28

#Hyperparameters

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

    normalize = transforms.Normalize(
        mean=0.1736,
        std=0.3248,
    )

    # define transform
    transform = transforms.Compose([
        transforms.ToTensor(), 
        normalize
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

def check_accuracy(loader, model):
    num_correct = 0 
    num_samples = 0 
    model.eval()

    with torch.no_grad(): 
        for x, y in loader: 
            x = x.to(device = device)
            y = y.to(device = device)
            #x = x.reshape(x.shape[0], -1)

            scores = model(x)
            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)

        print(f'Got {num_correct} / {num_samples} with accuracy {float(num_correct) / float(num_samples)*100:.2f}')
        return round((float(num_correct) / float(num_samples))*100, 2) 

#Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

torch.manual_seed(42)

train_loader, valid_loader = get_train_valid_loader(data_dir = './data', batch_size = 64, augment = False, random_seed = 1)

test_loader = get_test_loader(data_dir = './data', batch_size = 64)

model = My_CNN(imageDim,62).to(device)

# Loss and optimiser 
params = model.parameters()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(params=params, lr=lrng_rate, momentum = 0.9, weight_decay=wd)
#print(optimizer.param_groups)
#optimizer = optim.Adam(params=model.parameters(), lr=lrng_rate)

#Train network
print('Training the Deep Learning network ...')
#total_batch = len(mnist_train) // batch_size
#print('Size of the training dataset is {}'.format(mnist_train.data.size()))
#print('Size of the testing dataset'.format(mnist_test.data.size()))
#print('Batch size is : {}'.format(batch_size))
#print('Total number of batches is : {0:2.0f}'.format(total_batch))
#print('\nTotal number of epochs is : {0:2.0f}'.format(training_epochs))
counter_epoch = 0
for epoch in range(training_epochs):
    counter_epoch += 1
    print("Epoch n. ", counter_epoch)
    if (counter_epoch % 5) == 0:
        for g in optimizer.param_groups:
            print("Changing the learning rate from ", g['lr'])
            g['lr'] = g['lr'] * 0.1
            print("to ", g['lr'])
    model.train()
    for i, (data, targets) in enumerate(train_loader):
        #loading bar 
        n = len(train_loader)
        sys.stdout.write('\r')
        j = (i + 1) / n
        sys.stdout.write("[%-20s] %d%%" % ('='*int(20*j), 100*j))
        sys.stdout.flush()

        #Get data to cuda if possible 
        data = data.to(device = device)
        targets = targets.to(device = device)

        # forward 
        outputs = model(data)
        #print(np.shape(targets))
        #print(np.shape(outputs))
        #print(targets)
        #print(outputs)

        loss = criterion(outputs, targets)

        #backward 
        optimizer.zero_grad()
        loss.backward()

        #gradient descent or adam step 
        optimizer.step()
        
    # Validation
    #acc_tl = check_accuracy(train_loader, model)
    #acc_vl = check_accuracy(valid_loader, model)
    print("\n")
    model.eval()
    wandb.log({"acc_train": check_accuracy(train_loader, model), "acc_valid": check_accuracy(valid_loader, model), "loss": loss.item()})
    wandb.log({"acc_test": check_accuracy(test_loader, model)})

print('Learning Finished!')
wandb.finish()


    

