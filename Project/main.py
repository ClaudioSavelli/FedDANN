import os
import json
from collections import defaultdict
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


def model_init(args):
    if args.model == 'deeplabv3_mobilenetv2':
        return deeplabv3_mobilenetv2(num_classes=get_dataset_num_classes(args.dataset))
    if args.model == 'resnet18':
        model = resnet18()
        model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        model.fc = nn.Linear(in_features=512, out_features=get_dataset_num_classes(args.dataset))
        return model
    if args.model == 'cnn':
        return My_CNN(28*28,62)
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

        angles = [0, 15, 30, 45, 60, 75]
        out_angle = angles.pop( np.random.randint(len(angles)) )

        train_transforms = nptr.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            #transforms.rotate(np.random.choice(angles) if args.rotateFemnist else 0),
            normalize,
            #nptr.Normalize((0.5,), (0.5,))
        ])
        test_transforms = nptr.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            normalize,
            #nptr.Normalize((0.5,), (0.5,))
        ])
    else:
        raise NotImplementedError
    return train_transforms, test_transforms


def read_femnist_dir(data_dir):
    data = defaultdict(lambda: {})
    files = os.listdir(data_dir)
    files = [f for f in files if f.endswith('.json')]
    #files = np.random.choice(files, size = len(files)//4)
    i = 1
    for f in files:
        sys.stdout.write('\r')
        sys.stdout.write("%d / %d" % (i, len(files)))
        sys.stdout.flush()
        file_path = os.path.join(data_dir, f)
        
        with open(file_path, 'r') as inf:
            cdata = json.load(inf)
        data.update(cdata['user_data'])
        i += 1
    return data


def my_read_femnist_dir(data_dir, transform):
    data = []
    files = os.listdir(data_dir)
    files = [f for f in files if f.endswith('.json')]
    #files = np.random.choice(files, size = len(files)//4)
    i = 1
    for f in files:
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


def read_femnist_data(train_data_dir, test_data_dir):
    return read_femnist_dir(train_data_dir), read_femnist_dir(test_data_dir)

def my_read_femnist_data(train_data_dir, test_data_dir, train_transform, test_transform):
    return my_read_femnist_dir(train_data_dir, train_transform), my_read_femnist_dir(test_data_dir, test_transform)


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
        #train_data, test_data = read_femnist_data(train_data_dir, test_data_dir)
        train_datasets, test_datasets = my_read_femnist_data(train_data_dir, test_data_dir, train_transforms, test_transforms)

        #train_datasets, test_datasets = [], []

        #for user, data in train_data.items():
        #    train_datasets.append(Femnist(data, train_transforms, user))
        #for user, data in test_data.items():
        #    test_datasets.append(Femnist(data, test_transforms, user))

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
        # start a new wandb run to track this script
    parser = get_parser()
    args = parser.parse_args()
    set_seed(args.seed)

    wandb.init(
        mode="disabled",

        # set the wandb project where this run will be logged
        project="Femnist part 1",
        
        # track hyperparameters and run metadata
        
        config={
        "learning_rate": args.lr,
        "batch size": args.bs,
        "weight decay": args.wd,
        "momentum": args.m,
        "clipping gradient": args.clip, 
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
        "p": 0.5,
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
    train_datasets, test_datasets = get_datasets(args)
    print('\nDone.')

    metrics = set_metrics(args)
    train_clients, test_clients = gen_clients(args, train_datasets, test_datasets, model, device)
    #print("somma img train = ", sum([len(x) for x in train_datasets]))

    server = Server(args, train_clients, test_clients, model, metrics)
    server.train(args)

    wandb.finish()

if __name__ == '__main__':
    main()



