import numpy as np
import os
import sys
import matplotlib.pyplot as plt
import copy
from PIL import Image

sys.path.append("./datasets")

from datasets.femnist import Femnist

import json
import random
import torch
import torchvision.transforms as transforms

import datasets.ss_transforms as sstr
import datasets.np_transforms as nptr

total_clients = 1002

def set_seed(random_seed):
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def my_read_femnist_dir(data_dir, transform):
    data = []
    files = os.listdir(data_dir)
    files = [f for f in files if f.endswith('.json')]
    random.shuffle(files)

    n_file = 0
    n_clients = 0
    n_clients_per_angle = total_clients // 6
    while len(data) < total_clients:
        file_path = os.path.join(data_dir, files[n_file])
        with open(file_path, 'r') as inf:
            cdata = json.load(inf)
            for user, images in cdata['user_data'].items():
                if len(data) >= total_clients:
                    break
                # print(n_clients, "->", n_clients//n_clients_per_angle)
                data.append(Femnist(images, transform[n_clients//n_clients_per_angle], user))
                n_clients += 1
        n_file += 1

    data = data[:total_clients]

    return data


def my_read_femnist_data(train_data_dir, train_transform):
    return my_read_femnist_dir(train_data_dir, train_transform)

def get_transforms():
    angles = [0, 15, 30, 45, 60, 75]

    myAngleTransforms = []
    for theta in angles:
        t = transforms.Compose([
            transforms.ToPILImage(),
            # transforms.Lambda(lambda x: transforms.functional.rotate(x, angle=theta, fill=[0])),
            transforms.RandomRotation(degrees=(theta, theta), fill=(1,)),
            transforms.ToTensor(),
            #normalize,
        ])
        myAngleTransforms.append(copy.deepcopy(t))

    return myAngleTransforms


def get_clients():
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

    angle_counter = 0
    for i, c in enumerate(clients):
        angle_index = total_clients//6

        client_array.append(c.client_name)
        num_images_array.append(len(c))

        this_client_images = []
        this_client_labels = []
        for img, label in c:
            this_client_images.append(img.flatten().tolist())
            this_client_labels.append(label)

        images_array.append(this_client_images)
        labels_array.append(this_client_labels)
        
        if (i+1) % angle_index == 0:
            myDict = {}
            myDict["users"] = client_array
            myDict["numSamples"] = num_images_array
            myDict["user_data"] = {}

            for i,c in enumerate(client_array):
                myDict["user_data"][c] = {}
                myDict["user_data"][c]["x"] = images_array[i]
                myDict["user_data"][c]["y"] = labels_array[i]

            with open(f"data/RotatedFEMNIST/{angles[angle_counter]}.json", "w") as outfile:
                json.dump(myDict, outfile)
                
            angle_counter += 1

            client_array = []
            num_images_array = []
            images_array = []
            labels_array = []
            
def visualize(file):
    file_path = f"./data/RotatedFEMNIST/{file}.json"
    print(file_path)
    with open(file_path, 'r') as inf:
        cdata = json.load(inf)
        userdata = random.choice(list(cdata['user_data'].items()))
        index_ = random.randint(0, len(userdata[1]["x"]))
        img = userdata[1]["x"][index_]
        img = np.array(img).reshape(28,28)
        print(userdata[1]["y"][index_])
        plt.imshow(img, cmap="gray")
        plt.show()

if __name__ == "__main__":
    buildRotatedFEMNIST()
    # visualize(15)
    # visualize(30)
    # visualize(45)
    # visualize(60)
    # visualize(75)
