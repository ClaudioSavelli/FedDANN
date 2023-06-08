import torch 
import torch.distributions as distributions
import torch.nn.functional as F

from utils import *

imageDim = 28*28

class GradReverse(torch.autograd.Function):
    """Extension of grad reverse layer."""
    @staticmethod
    def forward(ctx, x):
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        grad_output = grad_output.neg()
        return grad_output, None

    def grad_reverse(x):
        return GradReverse.apply(x)

# Create Fully Connected Network
class DANN(torch.nn.Module):
    def __init__(self, input_size, num_classes, num_domains):
        super(DANN, self).__init__()

        self.num_classes = num_classes
        self.num_domains = num_domains

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
            torch.nn.ReLU()
        )
        
        # Second fully connected layer that outputs our 62 labels
        self.fc2 = torch.nn.Linear(2048, num_classes)

        self.dropout1 = torch.nn.Dropout(0.25)
        self.dropout2 = torch.nn.Dropout(0.25)
        self.dropout3 = torch.nn.Dropout(0.25)




        ### Domain Regressor
        self.fc1_dann = torch.nn.Sequential(
            torch.nn.Linear(2048, 1024),
            torch.nn.ReLU()
        )


        self.fc2_dann = torch.nn.Sequential(
            torch.nn.Linear(1024, num_domains),
            #torch.nn.Softmax(dim=-1)
        )

        
    def featurise(self, x):
        out = self.dropout1(self.layer1(x))
        out = self.dropout2(self.layer2(out))
        out = out.view(out.shape[0],-1)
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


    def domain_regressor(self, x):
        out = self.fc1_dann(x)
        out = self.fc2_dann(out)

        if out.isnan().any():
            for name, param in self.named_parameters():
                if param.requires_grad:
                    print ("name: ", name, "\ndatashape: ", param.data.shape, "\nisNaN: ", param.data.isnan().any(), "\n\n")
            input("press enter to continue.")

        return out

    def forward(self, x):         
        x = self.featurise(x)
        y = GradReverse.grad_reverse(x)
        return self.cls(x), self.domain_regressor(y)
