import numpy as np
import datasets.np_transforms as tr

import torchvision.transforms as transforms

from typing import Any
from torch.utils.data import Dataset

IMAGE_SIZE = 28

'''
transform = transforms.Compose([
     transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
 ])
 '''


class Femnist(Dataset):

    def __init__(self,
                 data: dict,
                 transform: tr.Compose,
                 client_name: str):
        super().__init__()
        self.samples = [[image, label] for image, label in zip(data['x'], data['y'])]
        self.transform = transform
        self.client_name = client_name
        self.domain = None

    def __getitem__(self, index: int) -> Any:
        img, label = self.samples[index]
        img = np.array(img, dtype=np.float32).reshape((28,28))
        img = self.transform(img)
        return img, label

    def __len__(self) -> int:
        return len(self.samples)

    def set_domain(self, domain):
        self.domain = domain
