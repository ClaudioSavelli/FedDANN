import numpy as np
import torch 
import torch.nn as nn 
import torch.optim as optim 
from torchvision import datasets
import torchvision.transforms as transforms
from models.cnn1 import My_CNN
from utils.stream_metrics import StreamClsMetrics

from utils.args import get_parser
import sys

import wandb

imageDim = 28*28

def set_seed(random_seed):
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def set_metrics():
    num_classes = 62
    
    metrics = {
            'eval_train': StreamClsMetrics(num_classes, 'eval_train'),
            'test': StreamClsMetrics(num_classes, 'test')
    }
    
    return metrics

@staticmethod
def update_metric(metric, outputs, labels):
    _, prediction = outputs.max(dim=1)
    labels = labels.cpu().numpy()
    prediction = prediction.cpu().numpy()
    metric.update(labels, prediction)

def get_train_valid_loader(data_dir,
                           batch_size,
                           valid_size=0.1,
                           shuffle=True):

    normalize = transforms.Normalize(
        mean=0.1736,
        std=0.3248,
    )

    # define transforms
    train_transform = transforms.Compose([
            transforms.ToTensor(),
            normalize
        ])

    # load the dataset
    train_set = datasets.EMNIST(
        root=data_dir, train=True,split = 'byclass',
        download=True, transform=train_transform,
    )

    entire_trainset = torch.utils.data.DataLoader(train_set, shuffle=shuffle)

    split_train_size = int((1-valid_size)*(len(entire_trainset))) 
    split_valid_size = len(entire_trainset) - split_train_size 

    train_set, val_set = torch.utils.data.random_split(train_set, [split_train_size, split_valid_size]) 

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=shuffle)
    valid_loader = torch.utils.data.DataLoader(val_set, batch_size=1, shuffle=shuffle)

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
        dataset, batch_size=1, shuffle=shuffle
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

            outputs = model(img)
            update_metric(metric, outputs, labels)


def main(): 
    parser = get_parser()
    args = parser.parse_args()
    metrics = set_metrics()

    p = 0.25
    mode_selected = "disabled" if args.test_mode else "online"

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
        "batch_size": args.bs,
        "Optimiser": "SGD",
        "criterion": "nn.CrossEntropyLoss()",
        "p": p,
        "lr interval modifier multiplying by 0.1": args.change_lr_interval,
        "seed": args.seed,
        "weight decay": args.wd
        }
    )

    set_seed(args.seed)

    train_loader, valid_loader = get_train_valid_loader(data_dir = './data', batch_size = args.bs)

    test_loader = get_test_loader(data_dir = './data', batch_size = args.bs)

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
        if (counter_epoch % args.change_lr_interval) == 0:
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
        model.eval()
        
        check_accuracy(valid_loader, model, metrics["eval_train"])

        #To put data on wandb
        results = metrics['eval_train'].get_results()
        for k, v in results.items():
            if k != 'Class Acc': 
                name = k + '_validation'
                wandb.log({name: v, "n_epoch": counter_epoch})
        print(metrics['eval_train'])

    check_accuracy(test_loader, model, metrics["test"])

    #To put data on wandb
    results = metrics['test'].get_results()
    for k, v in results.items():
        if k != 'Class Acc': 
            name = k + '_test'
            wandb.log({name: v})
    print(metrics['test'])


    print('Learning Finished!')
    wandb.finish()

if __name__ == '__main__':
    #Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    main()
    

