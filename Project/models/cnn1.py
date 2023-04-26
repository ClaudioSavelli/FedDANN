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

imageDim = 28*28

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

        self.dropout = nn.Dropout(0.5)

    def forward(self, x): 
        out = self.dropout(self.layer1(x))
        out = self.dropout(self.layer2(out))
        out = out.reshape(out.shape[0],-1)
        #print('x_shape:',out.shape)
        out = self.dropout(self.fc1(out))
        out = self.fc2(out) #to ask 
        return out

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