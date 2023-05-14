import numpy as np
import torch 
import torch.nn as nn 
import torch.optim as optim 
from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
from models.cnn1 import My_CNN
from utils.stream_metrics import StreamClsMetrics

from utils.args import get_parser
import sys

import wandb

imageDim = 28*28

def set_metrics():
    num_classes = 62
    
    metrics = {
            'eval_train': StreamClsMetrics(num_classes, 'eval_train'),
            'test': StreamClsMetrics(num_classes, 'test')
    }
    
    return metrics

parser = get_parser()
args = parser.parse_args()
metrics = set_metrics()

p = 0.25
mode_selected = "disabled" if args.test_mode else "online"


# start a new wandb run to track this script
wandb.init(
    mode=mode_selected,
    # set the wandb project where this run will be logged
    project="RealEmnistBenchmark",
    
    # track hyperparameters and run metadata
    config={
    "learning_rate": args.lr,
    "architecture": "CNN",
    "dataset": "EMNIST",
    "epochs": args.num_epochs,
    "momentum": args.m, 
    "Optimiser": "SGD",
    "criterion": "nn.CrossEntropyLoss()",
    "p": p,
    #"lr modifier": "Multiplied by 0.1 every 5 iterations",
    "seed": 42,
    "weight decay": args.wd
    }
)

@staticmethod
def update_metric(metric, outputs, labels):
    _, prediction = outputs.max(dim=1)
    labels = labels.cpu().numpy()
    prediction = prediction.cpu().numpy()
    metric.update(labels, prediction)

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

'''
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
'''



def check_accuracy(loader, model, metric):

    metric.reset()

    with torch.no_grad(): 
        for img, labels in loader: 
            img = img.to(device = device)
            labels = labels.to(device = device)
            #x = x.reshape(x.shape[0], -1)

            outputs = model(img)
            update_metric(metric, outputs, labels)





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
optimizer = optim.SGD(params=params, lr=args.lr, momentum = args.m, weight_decay=args.wd)


#Train network
print('Training the Deep Learning network ...')

counter_epoch = 0
for epoch in range(args.num_epochs):
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

        loss = criterion(outputs, targets)

        #backward 
        optimizer.zero_grad()
        loss.backward()

        #gradient descent or adam step 
        optimizer.step()
        
    # Validation

    print("\n")
    model.eval()
    
    check_accuracy(valid_loader, model, metrics["eval_train"])
    results = metrics['eval_train'].get_results()
    for k, v in results.items():
        if k != 'Class Acc': 
            name = k + '_validation'
            wandb.log({name: v})
    print(metrics['eval_train'])

check_accuracy(test_loader, model, metrics["test"])
results = metrics['test'].get_results()
for k, v in results.items():
    if k != 'Class Acc': 
        name = k + '_test'
        wandb.log({name: v})
print(metrics['test'])


print('Learning Finished!')
wandb.finish()


    

