import torch 
import torch.nn as nn 
import torch.optim as optim 
import torch.nn.functional as F 
from torch.utils.data import DataLoader 
import torchvision.transforms as transforms
import sys
from math import sqrt

imageDim = 28*28

#Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

torch.manual_seed(42)

# Create Fully Connected Network
class My_CNN(nn.Module): 
    def __init__(self, input_size, num_classes): 
        super(My_CNN, self).__init__()
        # First 2D convolutional layer, taking in 1 input channel (image),
        # outputting 32 convolutional features, with a square kernel size of 5
        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(1, 32, kernel_size=5),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2)) #32*25*25
        
        # Second 2D convolutional layer, taking in 1 input channel (image),
        # outputting 32 convolutional features, with a square kernel size of 5
        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(32, 64, kernel_size=5),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2)) #32*21*21

        # First fully connected layer
        self.fc1 = nn.Linear(32*21*21, 2048)
        # Second fully connected layer that outputs our 10 labels
        self.fc2 = nn.Linear(2048, num_classes)

    def forward(self, x): 
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.fc1(out)
        out = self.fc2(out)
        return out


model = My_CNN(imageDim,62).to(device)
model

#Hyperparameters
lrng_rate = 4*sqrt(2)
batch_size = 5
training_epochs = 3

# Loss and optimiser 
criterion = nn.CrossEntropyLoss()
optimzr = optim.Adam(params=model.parameters(), lr=lrng_rate) #???

#Train network
#print('Training the Deep Learning network ...')
#total_batch = len(mnist_train) // batch_size
#print('Size of the training dataset is {}'.format(mnist_train.data.size()))
#print('Size of the testing dataset'.format(mnist_test.data.size()))
#print('Batch size is : {}'.format(batch_size))
#print('Total number of batches is : {0:2.0f}'.format(total_batch))
#print('\nTotal number of epochs is : {0:2.0f}'.format(training_epochs))

for epoch in range(training_epochs):
    avg_cost = 0
    for i, (batch_X, batch_Y) in enumerate(data_loader):
        #loading bar 
        ##n = len(data_loader)
        ##sys.stdout.write('\r')
        ##j = (i + 1) / n
        ##sys.stdout.write("[%-20s] %d%%" % ('='*int(20*j), 100*j))
        ##sys.stdout.flush()

        #Get data to cuda if possible 
        batch_x = data.to(device = device)
        batch_y = targets.to(device = device)

        # forward 
        scores = model(data)
        loss = criterion(batch_X, batch_Y)

        #backward 
        optimizer.zero_grad()
        loss.backward()

        #gradient descent or adam step 
        optimizer.step() 

print('Learning Finished!')

def check_accuracy(loader, model):
    num_correct = 0 
    num_samples = 0 
    model.eval()

    with torch.no_grad(): 
        for x, y in loader: 
            x = x.to(device = device)
            y = y.to(device = device)
            x = x.reshape(x.shape[0], -1)

            scores = model(x)
            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)

        print(f'Got {num_correct} / {num_samples} with accuracy {float(num_correct) / float(num_samples)*100:.2f}')

