import numpy as np
import torch 
import torch.nn as nn 
import torch.optim as optim 
from torchvision import datasets
import torchvision.transforms as transforms
from models.cnn1 import My_CNN
from utils.stream_metrics import StreamClsMetrics

from datasets.femnist import Femnist

from utils.args import get_parser
import sys
import os 
import json

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
    valid_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=shuffle)

    return (train_loader, valid_loader)

def read_rotated_emnist_dir():
        data_dir = os.path.join('data', 'RotatedFEMNIST')
        
        normalize = transforms.Normalize(
        mean=0.1736,
        std=0.3248,
    )

        # define transforms
        transform = transforms.Compose([
            transforms.ToTensor(),
            normalize
        ])

        all_data = {}
        all_data['x'] = []
        all_data['y'] = []
        files = os.listdir(data_dir)
        files = [f for f in files if f.endswith('.json')]

        i = 1
        for f in files:
            #Loading bar
            sys.stdout.write('\r')
            sys.stdout.write("%d / %d" % (i, len(files)))
            sys.stdout.flush()
            file_path = os.path.join(data_dir, f)
            
            with open(file_path, 'r') as inf:
                cdata = json.load(inf)
                #t = 0 
                for user, images in cdata['user_data'].items():   
                    all_data['x'] += images["x"]
                    all_data['y'] += images["y"]
            i += 1
        
        return Femnist(all_data, transform, "Centralised User")

def split_train_test(entire_dataset, batch_size, train_size = 0.8, shuffle = True):
    split_train_size = int((train_size)*(len(entire_dataset))) 
    split_test_size = len(entire_dataset) - split_train_size 

    train_set, test_set = torch.utils.data.random_split(entire_dataset, [split_train_size, split_test_size])

    split_train_size = int((train_size)*(len(train_set))) 
    split_valid_size = len(train_set) - split_train_size 
    train_set, valid_set = torch.utils.data.random_split(train_set, [split_train_size, split_valid_size])


    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=shuffle)
    validation_loader = torch.utils.data.DataLoader(valid_set, batch_size=batch_size, shuffle=shuffle)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=shuffle)

    return (train_loader, validation_loader, test_loader)

def read_l1o_emnist_dir(leftout):
    data_dir = os.path.join('data', 'RotatedFEMNIST')
        
    normalize = transforms.Normalize(
        mean=0.1736,
        std=0.3248,
    )

    # define transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        normalize
    ])

    train_data = {}
    train_data['x'] = []
    train_data['y'] = []
    test_data = {}
    test_data['x'] = []
    test_data['y'] = []
    
    files = os.listdir(data_dir)
    files = [f for f in files if f.endswith('.json')]

    i = 1
    num_file = 0
    for f in files:
        #Loading bar
        sys.stdout.write('\r')
        sys.stdout.write("%d / %d" % (i, len(files)))
        sys.stdout.flush()
        file_path = os.path.join(data_dir, f)
        
        with open(file_path, 'r') as inf:
            cdata = json.load(inf)
            #t = 0 
            for user, images in cdata['user_data'].items(): 
                if(num_file != leftout):
                    train_data['x'] += images["x"]
                    train_data['y'] += images["y"]
                else: 
                    test_data['x'] += images["x"]
                    test_data['y'] += images["y"]

        i += 1
        num_file += 1
    
    return Femnist(train_data, transform, "Centralised Train User"), \
            Femnist(test_data, transform, "Centralised Test User")

def create_l1o_loader(train_dataset, test_dataset, batch_size, train_size = 0.8, shuffle = True): 
    split_train_size = int((train_size)*(len(train_dataset))) 
    split_valid_size = len(train_dataset) - split_train_size 
    train_set, valid_set = torch.utils.data.random_split(train_dataset, [split_train_size, split_valid_size])
    test_set = test_dataset


    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=shuffle)
    validation_loader = torch.utils.data.DataLoader(valid_set, batch_size=batch_size, shuffle=shuffle)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=shuffle)

    return (train_loader, validation_loader, test_loader)


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

    if args.dataset_selection == 'rotated': 
        name = f"{args.dataset_selection}_cr{args.clients_per_round}_epochs{args.num_epochs}_lr{args.lr}"
    elif args.dataset_selection == 'L1O': 
        name = f"{args.dataset_selection}_leftout{args.leftout}_cr{args.clients_per_round}_epochs{args.num_epochs}_lr{args.lr}"

    wandb.init(
        mode=mode_selected,

        # set the wandb project where this run will be logged
        project="NewRotatedEmnistBenchmark",
        name = name, 
        
        # track hyperparameters and run metadata
        config={
        "learning_rate": args.lr,
        "architecture": "CNN",
        "dataset": "EMNIST",
        "epochs": args.num_epochs,
        "momentum": args.m, 
        "batch_size": args.bs,
        "train_fraction": args.tf,
        "Optimiser": "SGD",
        "criterion": "nn.CrossEntropyLoss()",
        "p": p,
        "lr interval modifier multiplying by 0.1": args.change_lr_interval,
        "seed": args.seed,
        "weight decay": args.wd
        }
    )

    set_seed(args.seed)

    if args.dataset_selection == 'rotated': 
        dataset = read_rotated_emnist_dir()

        train_loader, validation_loader, test_loader = split_train_test(dataset, args.bs, train_size=args.tf)
    
    elif args.dataset_selection == 'L1O': 
        train_dataset, test_dataset = read_l1o_emnist_dir(args.leftout)
        train_loader, validation_loader, test_loader = create_l1o_loader(train_dataset, test_dataset, args.bs, train_size=args.tf)

    model = My_CNN(imageDim,62).to(device)

    # Loss and optimiser 
    params = model.parameters()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(params=params, lr=args.lr, momentum = args.m, weight_decay=args.wd)

    #Train network
    print('\nTraining the Deep Learning network ...')

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
        
        check_accuracy(validation_loader, model, metrics["eval_train"])

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
    

