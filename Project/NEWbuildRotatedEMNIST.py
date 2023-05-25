import numpy as np
import os
import sys
import matplotlib.pyplot as plt
import copy
from PIL import Image

sys.path.append("./datasets")

from datasets.femnist import Femnist

import shutil

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


def my_read_femnist_dir(data_dir, transform, files_list):
    data = []
    files = files_list[:11]

    for f in files: 
        print(f)

    n_clients = 0
    n_clients_per_angle = total_clients // 6
    for f in files:
        client_in_folder = 0
        file_path = os.path.join(data_dir, f)
        with open(file_path, 'r') as inf:
            cdata = json.load(inf)
            for user, images in cdata['user_data'].items():
                transform_to_do = n_clients//n_clients_per_angle
                if len(data) >= total_clients:
                    transform_to_do = 0
                # print(n_clients, "->", n_clients//n_clients_per_angle)
                data.append(Femnist(images, transform[transform_to_do], user))
                n_clients += 1
                client_in_folder += 1

            print(f, " ", client_in_folder)
    print("total clients taken from first phase: ", n_clients)
    #data = data[:total_clients]

    return data


def my_read_femnist_data(train_data_dir, train_transform, files_list):
    return my_read_femnist_dir(train_data_dir, train_transform, files_list)

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


def get_clients(files_list):
    train_datasets = []
    angle_transforms = get_transforms()
    train_data_dir = os.path.join('data', 'femnist', 'data', 'iid', 'train')
    clients = my_read_femnist_data(train_data_dir, angle_transforms, files_list)

    return clients

def buildRotatedFEMNIST(files_list):
    clients = get_clients(files_list)
    angles = [0, 15, 30, 45, 60, 75]

    client_array = []
    num_images_array = []
    images_array = []
    labels_array = []

    angle_counter = 0
    for i, c in enumerate(clients[:total_clients]):
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

            for j,c in enumerate(client_array):
                myDict["user_data"][c] = {}
                myDict["user_data"][c]["x"] = images_array[j]
                myDict["user_data"][c]["y"] = labels_array[j]

            with open(f"data/RotatedFEMNIST/{angles[angle_counter]}.json", "w") as outfile:
                json.dump(myDict, outfile)
                
            angle_counter += 1

            client_array = []
            num_images_array = []
            images_array = []
            labels_array = []

    print(len(clients))
    print(len(clients[total_clients:]))

    i = 0
    clients_left = len(clients)-total_clients
    print("number of clients left: ", clients_left)
    sp = total_clients
    ep = len(clients)
    #ep = sp + length
    '''
    while i != 16: 
        print(sp, " ", ep)
        for c in clients[sp:ep]:
            client_array.append(c.client_name)
            num_images_array.append(len(c))

            this_client_images = []
            this_client_labels = []
            for img, label in c:
                this_client_images.append(img.flatten().tolist())
                this_client_labels.append(label)

            images_array.append(this_client_images)
            labels_array.append(this_client_labels)
            
            
            myDict = {}
            myDict["users"] = client_array
            myDict["numSamples"] = num_images_array
            myDict["user_data"] = {}

            for j,c in enumerate(client_array):
                myDict["user_data"][c] = {}
                myDict["user_data"][c]["x"] = images_array[j]
                myDict["user_data"][c]["y"] = labels_array[j]

            with open(f"data/RotatedFEMNIST/rest_of_clients_{i}.json", "w") as outfile:
                json.dump(myDict, outfile)
        i = i + 1
        sp = sp + length
        ep = ep + length
    '''
    print(sp, " ", ep)
    for c in clients[sp:]:
            client_array.append(c.client_name)
            num_images_array.append(len(c))

            this_client_images = []
            this_client_labels = []
            for img, label in c:
                this_client_images.append(img.flatten().tolist())
                this_client_labels.append(label)

            images_array.append(this_client_images)
            labels_array.append(this_client_labels)
            
            
            myDict = {}
            myDict["users"] = client_array
            myDict["numSamples"] = num_images_array
            myDict["user_data"] = {}

            for j,c in enumerate(client_array):
                myDict["user_data"][c] = {}
                myDict["user_data"][c]["x"] = images_array[j]
                myDict["user_data"][c]["y"] = labels_array[j]

            with open(f"data/RotatedFEMNIST/rest_of_clients_0.json", "w") as outfile:
                json.dump(myDict, outfile)
            
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

def count_clients_in_file(file_path):
    print("Number of users in ", file_path)

    n_clients = 0

    with open(file_path, 'r') as inf:
        cdata = json.load(inf)
        for user, images in cdata['user_data'].items():
            n_clients += 1

    print(n_clients)

def copy_paste(file_list): 
    niid = False #args.niid
    data_dir = os.path.join('data', 'femnist', 'data', 'iid', 'train')
    data = []
    files = file_list[11:]

    i = 1
    for f in files:
        print(f)
        src = os.path.join(data_dir, f)
        dst = f"data/RotatedFEMNIST/rest_of_clients_{i}.json"
        shutil.copyfile(src, dst)
        i += 1
            

if __name__ == "__main__":
    set_seed(42)
    data_dir = os.path.join('data', 'femnist', 'data', 'iid', 'train')
    files = os.listdir(data_dir)
    files = [f for f in files if f.endswith('.json')]
    random.shuffle(files)
    buildRotatedFEMNIST(files)
    copy_paste(files)

    data_dir = os.path.join('data', 'RotatedFEMNIST')
    files = os.listdir(data_dir)
    files = [f for f in files if f.endswith('.json')]
    for f in files: 
        count_clients_in_file(os.path.join(data_dir, f))

    visualize(15)
    visualize(30)
    visualize(45)
    visualize(60)
    visualize(75)
    visualize("rest_of_clients_0")
    visualize("rest_of_clients_2")
