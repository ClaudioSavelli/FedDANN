import copy
import random
import numpy as np
import torch 
import torch.nn as nn 
import torch.optim as optim 
from torchvision import datasets
import torchvision.transforms as transforms
from models.cnn1 import My_CNN
from utils.stream_metrics import StreamClsMetrics

import datasets.np_transforms as nptr
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
            'test': StreamClsMetrics(num_classes, 'test'),
            'l1O': StreamClsMetrics(num_classes, 'l1O')
    }
    
    return metrics

@staticmethod
def update_metric(metric, outputs, labels):
    _, prediction = outputs.max(dim=1)
    labels = labels.cpu().numpy()
    prediction = prediction.cpu().numpy()
    metric.update(labels, prediction)

def get_transforms_rotated(args):
    
    normalize = transforms.Normalize(
    mean=0.1736,
    std=0.3248,
    )

    angles = [0, 15, 30, 45, 60, 75]
    myAngleTransforms = []
    for theta in angles:
        t = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomRotation(degrees=(theta, theta), fill=(1,)),
            transforms.ToTensor(),
            normalize,
        ])
        myAngleTransforms.append(copy.deepcopy(t))

    test_transforms = nptr.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        normalize,
    ])

    return myAngleTransforms, test_transforms

def apply_transforms(args, train_datasets, test_datasets):
    l1o_datasets = []

    ### FOR ROTATED
    if args.dataset_selection == 'rotated':
        train_transform_list, test_transforms = get_transforms_rotated(args)

        total_clients = 1002
        n_clients_per_angle = total_clients // 6
        for i, dataset in enumerate(train_datasets):
            transform_to_do = i // n_clients_per_angle
            dataset.set_transform(train_transform_list[ transform_to_do if i < total_clients else 0 ])

        for dataset in test_datasets:
            dataset.set_transform(test_transforms)

    ### FOR L1O
    elif args.dataset_selection == 'L1O':
        train_transform_list, test_transforms = get_transforms_rotated(args)

        total_clients = 1002
        n_clients_per_angle = total_clients // 6
        new_train_datasets = []

        for i, dataset in enumerate(train_datasets):
            transform_to_do = i // n_clients_per_angle
            dataset.set_transform(train_transform_list[ transform_to_do if i < total_clients else 0 ])

            if transform_to_do == args.leftout:
                l1o_datasets.append(dataset)
            else:
                new_train_datasets.append(dataset)
        
        for dataset in test_datasets:
            dataset.set_transform(test_transforms)

        train_datasets = new_train_datasets

    return train_datasets, test_datasets, l1o_datasets

def my_read_femnist_dir(data_dir, is_test_mode):
    data = []
    files = os.listdir(data_dir)
    files = [f for f in files if f.endswith('.json')]
    random.shuffle(files)
    if is_test_mode: files = np.random.choice(files, size = len(files)//12)

    i = 1
    for f in files:
        #Loading bar
        sys.stdout.write('\r')
        sys.stdout.write("%d / %d" % (i, len(files)))
        sys.stdout.flush()
        file_path = os.path.join(data_dir, f)
        
        with open(file_path, 'r') as inf:
            cdata = json.load(inf)
            for user, images in cdata['user_data'].items():    
                data.append(Femnist(images, None, user))
        i += 1
    
    return data

def unify_clients(dataset):
    all_data = {}
    all_data['x'] = []
    all_data['y'] = []
    i = 0

    for client in dataset: 
        list = client.get_list_of_samples()

        for img, label in list:
            all_data['x'] += img
            all_data['y'].append(label)
        i += 1
    
    return Femnist(all_data, None, "Centralised User")

def my_read_femnist_data(train_data_dir, test_data_dir, is_test_mode):
    return my_read_femnist_dir(train_data_dir, is_test_mode), \
           my_read_femnist_dir(test_data_dir, is_test_mode)

def get_datasets(args):

    train_datasets = []

    niid = args.niid
    train_data_dir = os.path.join('data', 'femnist', 'data', 'niid' if niid else 'iid', 'train')
    test_data_dir = os.path.join('data', 'femnist', 'data', 'niid' if niid else 'iid', 'test')
    train_datasets, test_datasets = my_read_femnist_data(train_data_dir, test_data_dir, args.test_mode)

    return train_datasets, test_datasets

def generate_data_loaders(train_dataset, test_dataset, batch_size, train_size = 0.8, shuffle = True):

    split_train_size = int((train_size)*(len(train_dataset))) 
    split_valid_size = len(train_dataset) - split_train_size 
    train_set, valid_set = torch.utils.data.random_split(train_dataset, [split_train_size, split_valid_size])

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=shuffle)
    validation_loader = torch.utils.data.DataLoader(valid_set, batch_size=batch_size, shuffle=shuffle)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle)

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
        project = "FinalRotatedFemnist"
        name = f"Emnist_{args.dataset_selection}"
    elif args.dataset_selection == 'L1O': 
        project = "FinalRotatedFemnist"
        name = f"Emnist_{args.dataset_selection}_leftout{args.leftout}"

    wandb.init(
        mode=mode_selected,

        # set the wandb project where this run will be logged
        project=project,
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
        "leftout": args.leftout, 
        "Optimiser": "SGD",
        "criterion": "nn.CrossEntropyLoss()",
        "p": p,
        "lr interval modifier multiplying by 0.1": args.change_lr_interval,
        "seed": args.seed,
        "weight decay": args.wd
        }
    )

    set_seed(args.seed)

    train_datasets, test_datasets = get_datasets(args)
    train_datasets, test_datasets, l1o_datasets = apply_transforms(args, train_datasets, test_datasets)

    train_datasets = unify_clients(train_datasets)
    test_datasets = unify_clients(test_datasets)

    train_loader, validation_loader, test_loader = generate_data_loaders(train_datasets, test_datasets, args.bs)

    if args.dataset_selection == 'L1O': 
        l1o_datasets = unify_clients(l1o_datasets)
        l1o_loader = torch.utils.data.DataLoader(l1o_datasets, batch_size=args.bs, shuffle=True)

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
            sys.stdout.write("[%-20s] %s%%" % ('=' * int(20 * j),  f"{(100 * j):.4}"))
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

    if args.dataset_selection == 'L1O': 
        check_accuracy(l1o_loader, model, metrics["l1O"])

    #To put data on wandb
    results = metrics['l1O'].get_results()
    for k, v in results.items():
        if k != 'Class Acc': 
            name = k + '_l1O'
            wandb.log({name: v})
    print(metrics['l1O'])


    print('Learning Finished!')
    wandb.finish()

if __name__ == '__main__':
    #Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    main()
    

