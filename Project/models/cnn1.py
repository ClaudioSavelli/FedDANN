import torch 
import torch.distributions as distributions
import torch.nn.functional as F

from utils import *

imageDim = 28*28

# Create Fully Connected Network
class My_CNN(torch.nn.Module): 
    def __init__(self, input_size, num_classes, is_prob=False):
        self.probabilistic = is_prob      # to be added to args
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
        
        # Second fully connected layer that outputs our 62 labels
        self.fc2 = torch.nn.Linear(2048, num_classes)

        self.dropout1 = torch.nn.Dropout(0.25)
        self.dropout2 = torch.nn.Dropout(0.25)
        self.dropout3 = torch.nn.Dropout(0.25)

        self.num_classes = num_classes
        
    def net(self, x): 
        out = self.dropout1(self.layer1(x))
        out = self.dropout2(self.layer2(out))
        out = out.reshape(out.shape[0],-1)
        out = self.dropout3(self.fc1(out))
        return out
    
    def cls(self, x):
        out = self.fc2(x)
        if out.isnan().any():
            for name, param in self.named_parameters():
                if param.requires_grad:
                    print ("name: ", name, "\ndatashape: ", param.data.shape, "\nisNaN: ", param.data.isnan().any(), "\n\n")
            input("press enter to continue.")
        return out
    
    def forward(self, x):         
        out = self.net(x)
        out = self.cls(out)

        #out = torch.nn.functional.softmax(out, dim = self.num_classes)
        return out
    
    def featurise(self, x, num_samples=1, return_dist=False):
        if not self.probabilistic:
            return self.net(x), (0.0, 0.0)
        else:
            z_params = self.net(x)
            z_mu = z_params[:,:]
            z_sigma = F.softplus(z_params[:,:])
            z_dist = distributions.Independent(distributions.normal.Normal(z_mu, z_sigma), 1)
            z = z_dist.rsample([num_samples]).view([-1, 2048])
            
            if return_dist:
                return z, (z_mu, z_sigma)
            else:
                return z, (0.0, 0.0)