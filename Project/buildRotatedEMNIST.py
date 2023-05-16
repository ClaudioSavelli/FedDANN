import numpy as np
import os
import sys
import json
import random
import torch
from datasets.femnist import Femnist
import torchvision.transforms as transforms

import datasets.ss_transforms as sstr
import datasets.np_transforms as nptr

def set_seed(random_seed):
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def my_read_femnist_dir(data_dir, transform, is_test_mode):
    data = []
    files = os.listdir(data_dir)
    files = [f for f in files if f.endswith('.json')]
    random.shuffle(files)

    n_file = 0
    n_clients = 0
    while len(data) < 1002:
        file_path = os.path.join(data_dir, files[n_file])
        with open(file_path, 'r') as inf:
            cdata = json.load(inf)
            for user, images in cdata['user_data'].items():
                data.append(Femnist(images, transform[ n_clients//6 ], user))
                n_clients += 1
        n_file += 1

    data = data[:1002]

    return data


def my_read_femnist_data(train_data_dir, train_transform):
    return my_read_femnist_dir(train_data_dir, train_transform)

def get_transforms():
    normalize = transforms.Normalize(
        mean=0.1736,
        std=0.3248,
    )
    angles = [0, 15, 30, 45, 60, 75]

    myAngleTransforms = []
    for theta in angles:
        t = nptr.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.rotate(theta),
            normalize,
        ])
        myAngleTransforms.append(t)

    return myAngleTransforms


def get_clients(args):
    train_datasets = []
    angle_transforms = get_transforms()
    niid = False #args.niid
    train_data_dir = os.path.join('data', 'femnist', 'data', 'niid' if niid else 'iid', 'train')
    clients = my_read_femnist_data(train_data_dir, angle_transforms)

    return clients

def buildRotatedFEMNIST():
    clients = get_clients()
    angles = [0, 15, 30, 45, 60, 75]

    client_array = []
    num_images_array = []
    images_array = []
    labels_array = []

    for i, c in enumerate(clients):
        angle_index = 1002//6

        client_array.append(c.client_name)
        num_images_array.append( len(c) )

        images_array.append([])
        labels_array.append([])
        for img, label in c:
            images_array[i].append(list(img.flatten()))
            labels_array[i].append(label)



        if (i-1) % angle_index == 0:
            myDict = {}
            myDict["users"] = client_array
            myDict["numSamples"] = num_images_array
            myDict["user_data"] = {}

            for i,c in enumerate(client_array):
                myDict["user_data"][c] = {}
                myDict["user_data"][c]["x"] = images_array[i]
                myDict["user_data"][c]["y"] = labels_array[i]

            with open(f"data/RotatedFEMNIST_{angles[angle_index]}.json", "w") as outfile:
                json.dump(myDict, outfile)

            client_array = []
            num_images_array = []
            images_array = []
            labels_array = []

if __name__ == "__main__":
    buildRotatedFEMNIST()
    