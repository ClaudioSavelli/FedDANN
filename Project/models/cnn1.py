import torch 
import torch.nn as nn 
import torch.optim as optim 
import torch.nn.functional as F 
import torchvision. datasets as dsets
from torch.utils.data import DataLoader 
import torchvision.transforms as transforms
import numpy as np
from math import sqrt

from utils import *

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

        self.num_classes = num_classes
    
    def forward(self, x): 
        print('1')
        #print(x)
        print(x.isnan().any())
        out = self.dropout(self.layer1(x))
        print('2')
        #print(out)
        print(out.isnan().any())
        out = self.dropout(self.layer2(out))
        print('3')
        #print(out)
        print(out.isnan().any())
        out = out.reshape(out.shape[0],-1)
        print('4')
        #print(out)
        print(out.isnan().any())
        #print('x_shape:',out.shape)
        out = self.dropout(self.fc1(out))
        print('5')
        #print(out)
        print(out.isnan().any())
        out = self.fc2(out) #to ask 
        print('6')
        #print(out)
        print(out.isnan().any())
        input("Press Enter to continue...")
        #out = torch.nn.functional.softmax(out, dim = self.num_classes)
        return out

def add_weight_decay(net, l2_value, skip_list=()): #https://raberrytv.wordpress.com/2017/10/29/pytorch-weight-decay-made-easy/
    decay, no_decay = [], []
    for name, param in net.named_parameters():
        if not param.requires_grad: continue # frozen weights		            
        if len(param.shape) == 1 or name.endswith(".bias") or name in skip_list: no_decay.append(param)
        else: decay.append(param)
    return [{'params': no_decay, 'weight_decay': 0.}, {'params': decay, 'weight_decay': l2_value}]

'''
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
        '''