import os
import json
import wandb
import sys
import copy
import matplotlib.pyplot as plt

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
from models.DANN import DANN
from models.fedSrNet import FedSrNet
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
    if args.model == 'fedsr':
        return FedSrNet(get_dataset_image_dimension(), get_dataset_num_classes(args.dataset), args)
    if args.model == 'dann':
        return DANN(get_dataset_image_dimension(), get_dataset_num_classes(args.dataset), 6)
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
    elif args.model == 'cnn' or args.model == 'resnet18' or args.model == 'fedsr' or args.model == 'dann':
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

def get_transforms_rotated(args):
    if args.model == 'cnn' or args.model == 'resnet18' or args.model == 'fedsr' or args.model == 'dann':
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
    else:
        raise NotImplementedError
    return myAngleTransforms, test_transforms

def apply_transforms(args, train_datasets, test_datasets):
    l1o_datasets = []

    ### FOR DEFAULT
    if args.dataset_selection == 'default':
        train_transforms, test_transforms = get_transforms(args)

        for dataset in train_datasets:
            dataset.set_transform(train_transforms)

        for dataset in test_datasets:
            dataset.set_transform(test_transforms)

    ### FOR ROTATED
    elif args.dataset_selection == 'rotated':
        train_transform_list, test_transforms = get_transforms_rotated(args)

        total_clients = 1002
        n_clients_per_angle = total_clients // 6
        for i, dataset in enumerate(train_datasets):
            transform_to_do = i // n_clients_per_angle
            dataset.set_transform(train_transform_list[ transform_to_do if i < total_clients else 0 ])
            dataset.set_domain(transform_to_do)

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
            dataset.set_domain(transform_to_do)

            if transform_to_do == args.leftout:
                l1o_datasets.append(dataset)
            else:
                new_train_datasets.append(dataset)
        
        for dataset in test_datasets:
            dataset.set_transform(test_transforms)

        train_datasets = new_train_datasets

    return train_datasets, test_datasets, l1o_datasets
            


def my_read_femnist_dir(data_dir, transform, is_test_mode):
    data = []
    files = os.listdir(data_dir)
    files = [f for f in files if f.endswith('.json')]
    random.shuffle(files)
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

def my_read_femnist_dir_rotated(data_dir, transform): #read all the files
    data = []
    files = os.listdir(data_dir)
    files = [f for f in files if f.endswith('.json')]

    num_file = 0
    for i, f in enumerate(files):
        #Loading bar
        sys.stdout.write('\r')
        sys.stdout.write("%d / %d" % (i, len(files)))
        sys.stdout.flush()
        file_path = os.path.join(data_dir, f)
        
        with open(file_path, 'r') as inf:
            cdata = json.load(inf)
            data.append([])
            for user, images in cdata['user_data'].items():    
                data[num_file].append(Femnist(images, transform, user))
                data[num_file][-1].set_domain(i if i < 6 else 0)
            num_file += 1

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

    #Normally it's 0.8, but considering that at the end we add 1000 clients manually in the train we re-evaluated the division number to keep it coherent
    const_division = 0.72

    train_datasets = []
    train_transforms, test_transforms = get_transforms(args)

    if args.dataset == 'femnist':
        full_data_dir = os.path.join('data', 'RotatedFEMNIST')
        full_datasets_lists = my_read_femnist_dir_rotated(full_data_dir, train_transforms)
        print("\nNumero di file roc: ", len(full_datasets_lists[6:]))
        print("Numero di file rotation: ", len(full_datasets_lists[:6]))

        ### Se dobbiamo solo far rotated
        if args.dataset_selection == 'rotated':
            all_data = []
            for domain in full_datasets_lists[6:]:
                all_data.extend(domain)
            
            random.shuffle(all_data)
            train_datasets = all_data[:int(len(all_data)*const_division)]
            test_datasets = all_data[int(len(all_data)*const_division):]

            for domain in full_datasets_lists[:6]: 
                train_datasets.extend(domain)
            
            random.shuffle(train_datasets)
            
        elif args.dataset_selection == 'L1O':
            all_data = []
            for domain in full_datasets_lists[6:]:
                all_data.extend(domain)
            
            random.shuffle(all_data)
            train_datasets = all_data[:int(len(all_data)*const_division)]
            test_datasets = all_data[int(len(all_data)*const_division):]

            for i, domain in enumerate(full_datasets_lists[:6]): 
                if i != args.leftout:
                    train_datasets.extend(domain)
            
            random.shuffle(train_datasets)

    else:
        raise NotImplementedError

    return train_datasets, test_datasets

def take_l1o_loader(args, model, device): 
    train_transforms, test_transforms = get_transforms(args)
    data_dir = os.path.join('data', 'RotatedFEMNIST')
    data = []
    clients = []
    files = os.listdir(data_dir)
    files = [f for f in files if f.endswith('.json')]
    f = files[args.leftout]
    print("\nFile leftover: ", f)
    file_path = os.path.join(data_dir, f)
    
    with open(file_path, 'r') as inf:
        cdata = json.load(inf)
        for user, images in cdata['user_data'].items():    
            data.append(Femnist(images, test_transforms, user))
        
        for ds in data:
            clients.append(Client(args, ds, model, test_client = 1, device=device))
    return clients

def set_metrics(args):
    num_classes = get_dataset_num_classes(args.dataset)
    if args.model == 'deeplabv3_mobilenetv2':
        metrics = {
            'eval_train': StreamSegMetrics(num_classes, 'eval_train'),
            'test_same_dom': StreamSegMetrics(num_classes, 'test_same_dom'),
            'test_diff_dom': StreamSegMetrics(num_classes, 'test_diff_dom')
        }
    elif args.model == 'resnet18' or args.model == 'cnn' or args.model == 'fedsr' or args.model == 'dann':
        metrics = {
            'eval_train': StreamClsMetrics(num_classes, 'eval_train'),
            'test': StreamClsMetrics(num_classes, 'test'), 
            'l1O': StreamClsMetrics(num_classes, 'l1O')
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


def initWandB(args):
    wandbConfig = {
        "learning_rate": args.lr,
        "batch size": args.bs,
        "weight decay": args.wd,        
        "momentum": args.m, 
        "server_momentum": args.sm,
        "seed": args.seed,
        "isNiid": args.niid,
        "dataset": args.dataset_selection,
        "model": args.model,
        "num_rounds": args.num_rounds,
        "num_local_epochs": args.num_epochs,
        "clients_per_round": args.clients_per_round,
        "client_selection": args.client_selection,
        "pow_d": args.pow_d, 
        "architecture": "CNN",
        "dataset": "FeMnist",
        "Optimiser": "SGD",
        "criterion": "CrossEntropyLoss",
        "p": 0.25,
        "l2r": args.l2r, 
        "cmi": args.cmi, 
        "z_dim": args.z_dim
        }


    if args.client_selection == 'biased':
        project = "Real_SmartClientSelection"
        name = f"{'niid' if args.niid else 'iid'}_{args.client_selection}_cr{args.clients_per_round}_epochs{args.num_epochs}_lr{args.lr}"
    
    elif args.client_selection == 'pow':
        project = "Real_SmartClientSelection"
        name = f"{'niid' if args.niid else 'iid'}_{args.client_selection}_{args.pow_first_selection}_cr{args.clients_per_round}_d{args.pow_d}_epochs{args.num_epochs}_lr{args.lr}"
        wandbConfig["pow_selection"] = args.pow_first_selection
        if args.pow_d > 10: project = "Real_SmartClientSelection_big_d"
    else:
        ## Data selection projects
        if args.dataset_selection == 'default': 
            project = "RealFemnist part 1"
            name = f"{'niid' if args.niid else 'iid'}_cr{args.clients_per_round}_epochs{args.num_epochs}_lr{args.lr}"
            if args.sm != 0: 
                project = "Server Momentum Femnist"
                name = f"{'niid' if args.niid else 'iid'}_sm{args.sm}_cr{args.clients_per_round}_epochs{args.num_epochs}_lr{args.lr}"
        elif args.dataset_selection == 'rotated': 
            if args.model == 'fedsr': 
                project = "FinalRotatedFemnist" 
                name = f"{args.dataset_selection}_{args.model}_l1r{args.l2r}_cmi{args.cmi}"
            elif args.model == 'dann':
                project = "FinalRotatedFemnist"
                name = f"{args.dataset_selection}_{args.model}_w{args.dann_w}"
            else:
                project = "FinalRotatedFemnist" 
                name = f"{args.dataset_selection}_{args.model}"
        elif args.dataset_selection == 'L1O':
            if args.model == 'fedsr':
                project = "FinalRotatedFemnist"
                name = f"{args.dataset_selection}_{args.model}_leftout{args.leftout}_l1r{args.l2r}_cmi{args.cmi}"
                wandbConfig["leftout"] = args.leftout
            elif args.model == 'dann':
                project = "FinalRotatedFemnist"
                name = f"{args.dataset_selection}_{args.model}_leftout{args.leftout}_w{args.dann_w}"
                wandbConfig["leftout"] = args.leftout
            else:
                project = "FinalRotatedFemnist" 
                name = f"{args.dataset_selection}_{args.model}_leftout{args.leftout}"
                wandbConfig["leftout"] = args.leftout
    #name = "l2regularizer_L1O_leftout0_cr5_epochs1_lr0.1"
    mode_selected = "disabled" if args.test_mode else "online"
    wandb.init(
        mode=mode_selected,

        # set the wandb project where this run will be logged
        project=project,
        name=name,
        # track hyperparameters and run metadata
        config=wandbConfig
    )


def main():
    parser = get_parser()
    args = parser.parse_args()
    set_seed(args.seed)

    initWandB(args)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)
    torch.manual_seed(args.seed)

    print(f'Initializing model...')
    model = model_init(args)
    model.to("cuda")
    print('Done.')

    print('Generate datasets... ')
    # if args.dataset_selection == 'default':
    #     train_datasets, test_datasets = get_datasets(args)
    # elif args.dataset_selection == 'rotated':
    #     train_datasets, test_datasets = get_datasets_rotated(args)
    # elif args.dataset_selection == 'L1O':
    #     train_datasets, test_datasets = get_datasets_rotated(args)
    # else:
    #     raise Exception("Wrong dataset selection.")
    train_datasets, test_datasets = get_datasets(args)
    print('\nDone.')

    print('Applying transformations... ')
    train_datasets, test_datasets, l1o_datasets = apply_transforms(args, train_datasets, test_datasets)
    print("\nDone.")

    metrics = set_metrics(args)

    print("Generating clients... ", end = "")
    train_clients, test_clients = gen_clients(args, train_datasets, test_datasets, model, device)
 
    print("Done.")

    print("Generating server... ", end="")
    server = Server(args, train_clients, test_clients, model, metrics)


    for i in range(6):
        if i == args.leftout: continue

        client = None
        while client == None:
            client = np.random.choice(train_clients)
            if client.dataset.domain != i:
                client = None

        i_ = np.random.randint(len(client.dataset))
        img, label = client.dataset[i_]
        img = np.array(img).reshape(28,28)
        print("domain",i, "label",label)
        plt.imshow(img, cmap="gray")
        plt.show()

    input()



    if args.dataset_selection == 'L1O': 
        _ , l1o_clients = gen_clients(args, [], l1o_datasets, model, device)
        server.set_l1O_clients(l1o_clients)
    print("Done.")

    server.train(args)  
    
    wandb.finish()

if __name__ == '__main__':
    main()



