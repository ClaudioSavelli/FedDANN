import torch 
import torch.nn as nn 
import torch.optim as optim 
import torch.nn.functional as F 
import torchvision. datasets as dsets
from torch.utils.data import DataLoader 
import torchvision.transforms as transforms
import numpy as np
import sys
from math import sqrt

from utils import *

import wandb
import random


# start a new wandb run to track this script
wandb.init(
    # set the wandb project where this run will be logged
    project="EmnistBenchmark",
    
    # track hyperparameters and run metadata
    config={
    "learning_rate": 0.01,
    "architecture": "CNN",
    "dataset": "EMNIST",
    "epochs": 20,
    "Optimiser": "SGD(params=model.parameters(), lr=lrng_rate, momentum = 0.9)",
    "criterion": "nn.CrossEntropyLoss()",
    "p": 0.5,
    "lr modifier": "Multiplied by 0.1 every 5 iterations",
    "seed": 42,
    "weight decay": 1e-4
    }
)

imageDim = 28*28

#Hyperparameters
lrng_rate = 0.01
training_epochs = 20
p = 0.5
wd = 1e-4

# Create Fully Connected Network
class My_CNN(nn.Module): 
    def __init__(self, input_size, num_classes): 
        super(My_CNN, self).__init__()
        # First 2D convolutional layer, taking in 1 input channel (image),
        # outputting 32 convolutional features, with a square kernel size of 5
        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(1, 32, kernel_size=5),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2)) #32*12*12
        
        # Second 2D convolutional layer, taking in 1 input channel (image),
        # outputting 32 convolutional features, with a square kernel size of 5
        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(32, 64, kernel_size=5),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2)) #64*4*4

        # First fully connected layer
        self.fc1 = nn.Linear(1024, 2048)
        # Second fully connected layer that outputs our 10 labels
        self.fc2 = nn.Linear(2048, num_classes)

        self.dropout = nn.Dropout(p)

    def forward(self, x): 
        out = self.dropout(self.layer1(x))
        out = self.dropout(self.layer2(out))
        out = out.reshape(out.shape[0],-1)
        #print('x_shape:',out.shape)
        out = self.dropout(self.fc1(out))
        out = self.fc2(out) #to ask 
        return out

def add_weight_decay(net, l2_value, skip_list=()): #https://raberrytv.wordpress.com/2017/10/29/pytorch-weight-decay-made-easy/
    decay, no_decay = [], []
    for name, param in net.named_parameters():
        if not param.requires_grad: continue # frozen weights		            
        if len(param.shape) == 1 or name.endswith(".bias") or name in skip_list: no_decay.append(param)
        else: decay.append(param)
    return [{'params': no_decay, 'weight_decay': 0.}, {'params': decay, 'weight_decay': l2_value}]

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
params = add_weight_decay(model, wd)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(params=params, lr=lrng_rate, momentum = 0.9)
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
    wandb.log({"acc_train": check_accuracy(train_loader, model), "acc_valid": check_accuracy(valid_loader, model), "loss": loss.item()})
    wandb.log({"acc_test": check_accuracy(test_loader, model)})

print('Learning Finished!')
wandb.finish()


    

