import os
import json
import wandb
import sys

import torch
import random

import numpy as np
from torchvision.models import resnet18

import datasets.ss_transforms as sstr
import datasets.np_transforms as nptr

import torchvision.transforms as transforms

from torch import nn
from client import Client
from datasets.femnist import Femnist
from server import Server
from utils.args import get_parser
from datasets.idda import IDDADataset
from models.deeplabv3 import deeplabv3_mobilenetv2
from models.cnn1 import My_CNN
from utils.stream_metrics import StreamSegMetrics, StreamClsMetrics


def set_seed(random_seed):
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def get_dataset_num_classes(dataset):
    if dataset == 'idda':
        return 16
    if dataset == 'femnist':
        return 62
    raise NotImplementedError

def get_dataset_image_dimension(): 
    return 28*28


def model_init(args):
    if args.model == 'deeplabv3_mobilenetv2':
        return deeplabv3_mobilenetv2(num_classes=get_dataset_num_classes(args.dataset))
    if args.model == 'resnet18':
        model = resnet18()
        model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        model.fc = nn.Linear(in_features=512, out_features=get_dataset_num_classes(args.dataset))
        return model
    if args.model == 'cnn':
        return My_CNN(get_dataset_image_dimension(), get_dataset_num_classes(args.dataset))
    raise NotImplementedError


def get_transforms(args):
    # TODO: test your data augmentation by changing the transforms here!
    if args.model == 'deeplabv3_mobilenetv2':
        train_transforms = sstr.Compose([
            sstr.RandomResizedCrop((512, 928), scale=(0.5, 2.0)),
            sstr.ToTensor(),
            sstr.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        test_transforms = sstr.Compose([
            sstr.ToTensor(),
            sstr.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    elif args.model == 'cnn' or args.model == 'resnet18':
        normalize = transforms.Normalize(
        mean=0.1736,
        std=0.3248,
        )
        
        train_transforms = nptr.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            normalize,
        ])
        test_transforms = nptr.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        raise NotImplementedError
    return train_transforms, test_transforms

def my_read_femnist_dir(data_dir, transform, is_test_mode):
    data = []
    files = os.listdir(data_dir)
    files = [f for f in files if f.endswith('.json')]
    # files = random.shuffle(files)
    if is_test_mode: files = np.random.choice(files, size = len(files)//6)

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
                data.append(Femnist(images, transform, user))
        i += 1
    
    return data


def my_read_femnist_dir_rotated(data_dir, transform):
    data = []
    files = os.listdir(data_dir)
    files = [f for f in files if f.endswith('.json')]

    i = 1
    for f in files:
        #Loading bar
        sys.stdout.write('\r')
        sys.stdout.write("%d / %d" % (i, len(files)))
        sys.stdout.flush()
        file_path = os.path.join(data_dir, f)
        
        num_file = 0
        with open(file_path, 'r') as inf:
            cdata = json.load(inf)
            data[num_file] = []
            for user, images in cdata['user_data'].items():    
                data[num_file].append(Femnist(images, transform, user))
            num_file += 1
        i += 1
    
    return data

def my_read_femnist_data(train_data_dir, test_data_dir, train_transform, test_transform, is_test_mode):
    return my_read_femnist_dir(train_data_dir, train_transform, is_test_mode), \
           my_read_femnist_dir(test_data_dir, test_transform, is_test_mode)


def get_datasets(args):

    train_datasets = []
    train_transforms, test_transforms = get_transforms(args)

    if args.dataset == 'idda':
        root = 'data/idda'
        with open(os.path.join(root, 'train.json'), 'r') as f:
            all_data = json.load(f)
        for client_id in all_data.keys():
            train_datasets.append(IDDADataset(root=root, list_samples=all_data[client_id], transform=train_transforms,
                                              client_name=client_id))
        with open(os.path.join(root, 'test_same_dom.txt'), 'r') as f:
            test_same_dom_data = f.read().splitlines()
            test_same_dom_dataset = IDDADataset(root=root, list_samples=test_same_dom_data, transform=test_transforms,
                                                client_name='test_same_dom')
        with open(os.path.join(root, 'test_diff_dom.txt'), 'r') as f:
            test_diff_dom_data = f.read().splitlines()
            test_diff_dom_dataset = IDDADataset(root=root, list_samples=test_diff_dom_data, transform=test_transforms,
                                                client_name='test_diff_dom')
        test_datasets = [test_same_dom_dataset, test_diff_dom_dataset]

    elif args.dataset == 'femnist':
        niid = args.niid
        train_data_dir = os.path.join('data', 'femnist', 'data', 'niid' if niid else 'iid', 'train')
        test_data_dir = os.path.join('data', 'femnist', 'data', 'niid' if niid else 'iid', 'test')
        train_datasets, test_datasets = my_read_femnist_data(train_data_dir, test_data_dir, train_transforms, test_transforms, args.test_mode)

    else:
        raise NotImplementedError

    return train_datasets, test_datasets


def get_datasets_rotated(args):

    train_datasets = []
    train_transforms, test_transforms = get_transforms(args)

    if args.dataset == 'femnist':
        full_data_dir = os.path.join('data', 'RotatedFEMNIST')
        full_datasets_lists = my_read_femnist_dir_rotated(full_data_dir, train_transforms)
        if args.dataset_selection == 'rotated':
            all_data = []
            for domain in full_datasets_lists:
                all_data.extend(domain)
            
            random.shuffle(all_data)
            train_datasets = all_data[:int(len(all_data)*0.8)]
            test_datasets = all_data[int(len(all_data)*0.8):]
            
        elif args.dataset_selection == 'L1O':
            test_datasets = full_datasets_lists[args.leftout]
            for i, domain in enumerate(full_datasets_lists):
                if i == args.leftout:
                    continue
                train_datasets.extend(domain)
            random.shuffle(train_datasets)
                
    else:
        raise NotImplementedError

    return train_datasets, test_datasets


def set_metrics(args):
    num_classes = get_dataset_num_classes(args.dataset)
    if args.model == 'deeplabv3_mobilenetv2':
        metrics = {
            'eval_train': StreamSegMetrics(num_classes, 'eval_train'),
            'test_same_dom': StreamSegMetrics(num_classes, 'test_same_dom'),
            'test_diff_dom': StreamSegMetrics(num_classes, 'test_diff_dom')
        }
    elif args.model == 'resnet18' or args.model == 'cnn':
        metrics = {
            'eval_train': StreamClsMetrics(num_classes, 'eval_train'),
            'test': StreamClsMetrics(num_classes, 'test')
        }
    else:
        raise NotImplementedError
    return metrics


def gen_clients(args, train_datasets, test_datasets, model, device):
    clients = [[], []]
    for i, datasets in enumerate([train_datasets, test_datasets]):
        for ds in datasets:
            clients[i].append(Client(args, ds, model, test_client=i == 1, device=device))
    return clients[0], clients[1]


def main():
    parser = get_parser()
    args = parser.parse_args()
    set_seed(args.seed)

    mode_selected = "disabled" if args.test_mode else "online"
    wandb.init(
        mode=mode_selected,

        # set the wandb project where this run will be logged
        project="RealFemnist part 1",
        name=f"{'niid' if args.niid else 'iid'}_cr{args.clients_per_round}_epochs{args.num_epochs}_lr{args.lr}",
        # track hyperparameters and run metadata
        config={
        "learning_rate": args.lr,
        "batch size": args.bs,
        "weight decay": args.wd,
        "momentum": args.m, 
        "seed": args.seed,
        "isNiid": args.niid,
        "model": args.model,
        "num_rounds": args.num_rounds,
        "num_local_epochs": args.num_epochs,
        "clients_per_round": args.clients_per_round,
        "client_selection": args.client_selection,
        "pow_d": args.d, 
        "architecture": "CNN",
        "dataset": "FeMnist",
        "Optimiser": "SGD",
        "criterion": "CrossEntropyLoss",
        "p": 0.25,
        }
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)
    torch.manual_seed(args.seed)

    print(f'Initializing model...')
    model = model_init(args)
    model.to("cuda")
    print('Done.')

    print('Generate datasets...')
    if args.dataset_selection == 'default':
        train_datasets, test_datasets = get_datasets(args)
    elif args.dataset_selection == 'rotated':
        train_datasets, test_datasets = get_datasets(args)
    elif args.dataset_selection == 'L1O':
        train_datasets, test_datasets = get_datasets(args)
    else:
        raise Exception("Wrong dataset selection.")
        
    print('\nDone.')
    #Per data aug fare in modo che ci sia un train_dataset e un test_dataset come lo voglio io 

    metrics = set_metrics(args)
    train_clients, test_clients = gen_clients(args, train_datasets, test_datasets, model, device)

    server = Server(args, train_clients, test_clients, model, metrics)
    server.train(args)

    wandb.finish()

if __name__ == '__main__':
    main()



