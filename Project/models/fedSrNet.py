import torch 
from torch import nn
import torch.distributions as distributions
import torch.nn.functional as F

from utils import *
from models.cnn1 import My_CNN

imageDim = 28*28

class SqueezeLastTwo(nn.Module):
    """A module which squeezes the last two dimensions, ordinary squeeze can be a problem for batch size 1"""
    def __init__(self):
        super(SqueezeLastTwo, self).__init__()

    def forward(self, x):
        return x.view(x.shape[0], -1)

class FedSrNet(torch.nn.Module):
    def __init__(self, input_size, num_classes, args):
        super(FedSrNet, self).__init__()
        
        self.prob = args.prob
        self.z_dim = args.z_dim
        self.out_dim = 2*self.z_dim if self.prob else self.z_dim
        self.num_classes = num_classes
        
        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(1, 32, kernel_size=5),
            #torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2),
            torch.nn.Dropout(0.25)) #32*12*12
        
        # Second 2D convolutional layer, taking in 1 input channel (image),
        # outputting 32 convolutional features, with a square kernel size of 5
        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(32, 64, kernel_size=5),
            #torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2),
            torch.nn.Dropout(0.25),
            SqueezeLastTwo()) #64*4*4

        # First fully connected layer
        self.fc1 = torch.nn.Sequential(
            torch.nn.Linear(1024, 2048), 
            torch.nn.ReLU(),
            torch.nn.Dropout(0.25))
        
        self.fc2 = torch.nn.Sequential(
            torch.nn.Linear(2048, self.out_dim), 
            torch.nn.ReLU(),
            torch.nn.Dropout(0.25))
        
        # Second fully connected layer that outputs our 62 labels
        self.fc3 = torch.nn.Linear(self.z_dim, self.num_classes)

        # self.dropout1 = torch.nn.Dropout(0.25)
        # self.dropout2 = torch.nn.Dropout(0.25)
        # self.dropout3 = torch.nn.Dropout(0.25)
        
        self.net = nn.Sequential(self.layer1,
                                 self.layer2,
                                 self.fc1,
                                 self.fc2)
        
        self.cls = self.fc3
        # self.cls = nn.Linear(1024, self.num_classes)

        self.net.to(args.device)
        self.cls.to(args.device)
        self.model = nn.Sequential(self.net, self.cls)
        
        
    def forward(self, x):
        if not self.prob:
            out = self.model(x)
        else:
            z, (_, _) = self.featurise(x)
            out = self.cls(z)
        
        if out.isnan().any():
            for name, param in self.named_parameters():
                if param.requires_grad:
                    print ("name: ", name, "\ndatashape: ", param.data.shape, "\nisNaN: ", param.data.isnan().any(), "\n\n")
            input("press enter to continue.")
        return out
        
        
    
    def featurise(self, x, num_samples=1, return_dist=False):
        if not self.prob:
            return self.net(x), (0.0, 0.0)
        else:
            z_params = self.net(x)
            z_mu = z_params[:, :self.z_dim]
            z_sigma = F.softplus(z_params[:, self.z_dim:])
            z_dist = distributions.Independent(distributions.normal.Normal(z_mu, z_sigma), 1)
            z = z_dist.rsample([num_samples]).view([-1, self.z_dim])
            
            if return_dist:
                return z, (z_mu, z_sigma)
            else:
                return z, (0.0, 0.0)