import torch 
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
class My_CNN(torch.nn.Module): 
    def __init__(self, input_size, num_classes): 
        super(My_CNN, self).__init__()
        # First 2D convolutional layer, taking in 1 input channel (image),
        # outputting 32 convolutional features, with a square kernel size of 5
        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(1, 32, kernel_size=5),
            #torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2)) #32*12*12
        
        # Second 2D convolutional layer, taking in 1 input channel (image),
        # outputting 32 convolutional features, with a square kernel size of 5
        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(32, 64, kernel_size=5),
            #torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2)) #64*4*4

        # First fully connected layer
        self.fc1 = torch.nn.Sequential(
            torch.nn.Linear(1024, 2048), 
            torch.nn.ReLU())
        
        # Second fully connected layer that outputs our 10 labels
        self.fc2 = torch.nn.Linear(2048, num_classes)

        self.dropout1 = torch.nn.Dropout(0.25)#fare un dropout per ogni layer e abbassare il valore a 0.25
        self.dropout2 = torch.nn.Dropout(0.25)
        self.dropout3 = torch.nn.Dropout(0.25)

        self.num_classes = num_classes
    
    def forward(self, x): 
        #print('1')
        #print(x)
        #print(x.isnan().any())
        if x.isnan().any():
            print('1')
            for name, param in self.named_parameters():
                if param.requires_grad:
                    print ("name: ", name, "\ndatashape: ", param.data.shape, "\nisNaN: ", param.data.isnan().any(), "\n\n")
            input("press enter to continue.")

        
        out = self.dropout1(self.layer1(x))
        #print('2')
        if out.isnan().any():
            print('2')
            for name, param in self.named_parameters():
                if param.requires_grad:
                    print ("name: ", name, "\ndatashape: ", param.data.shape, "\nisNaN: ", param.data.isnan().any(), "\n\n")
            input("press enter to continue.")
        #print(out)
        #print(out.isnan().any())


        out = self.dropout2(self.layer2(out))
        if out.isnan().any():
            print('3')
            for name, param in self.named_parameters():
                if param.requires_grad:
                    print ("name: ", name, "\ndatashape: ", param.data.shape, "\nisNaN: ", param.data.isnan().any(), "\n\n")
            input("press enter to continue.")
        #print('3')
        #print(out)
        #print(out.isnan().any())


        out = out.reshape(out.shape[0],-1)
        if out.isnan().any():
            print('4')
            for name, param in self.named_parameters():
                if param.requires_grad:
                    print ("name: ", name, "\ndatashape: ", param.data.shape, "\nisNaN: ", param.data.isnan().any(), "\n\n")
            input("press enter to continue.")
        #print('4')
        #print(out)
        #print(out.isnan().any())
        #print('x_shape:',out.shape)


        out = self.dropout3(self.fc1(out))
        if out.isnan().any():
            print('5')
            for name, param in self.named_parameters():
                if param.requires_grad:
                    print ("name: ", name, "\ndatashape: ", param.data.shape, "\nisNaN: ", param.data.isnan().any(), "\n\n")
            input("press enter to continue.")
        #print('5')
        #print(out)
        #print(out.isnan().any())


        out = self.fc2(out) #to ask 
        if out.isnan().any():
            print('6')
            for name, param in self.named_parameters():
                if param.requires_grad:
                    print ("name: ", name, "\ndatashape: ", param.data.shape, "\nisNaN: ", param.data.isnan().any(), "\n\n")
            input("press enter to continue.")
        #print('6')
        #print(out)
        #if(out.isnan().any()):
        #    print(out.isnan().any())
        #    input("Press Enter to continue...")
        #out = torch.nn.functional.softmax(out, dim = self.num_classes)
        return out

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