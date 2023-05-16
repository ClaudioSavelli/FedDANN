import copy
import torch

from torch import optim, nn
from torch.utils.data import DataLoader
import copy

from utils.utils import HardNegativeMining, MeanReduction


class Client:

    def __init__(self, args, dataset, model, test_client=False, device=None):
        self.args = args
        self.dataset = dataset
        self.name = self.dataset.client_name
        self.model = model
        self.train_loader = DataLoader(self.dataset, batch_size=self.args.bs, shuffle=True, drop_last=True) \
            if not test_client else None
        self.test_loader = DataLoader(self.dataset, batch_size=1, shuffle=False)

        self.criterion = nn.CrossEntropyLoss(ignore_index=255)
        self.reduction = HardNegativeMining() if self.args.hnm else MeanReduction()

        self.device = device

    def __str__(self):
        return self.name

    @staticmethod
    def update_metric(metric, outputs, labels):
        _, prediction = outputs.max(dim=1)
        labels = labels.cpu().numpy()
        prediction = prediction.cpu().numpy()
        metric.update(labels, prediction)

    def _get_outputs(self, images):
        if self.args.model == 'deeplabv3_mobilenetv2':
            return self.model(images)['out']
        elif self.args.model == 'resnet18':
            return self.model(images)
        elif self.args.model == 'cnn': 
            return self.model(images)
        raise NotImplementedError

    def run_epoch(self, cur_epoch, optimizer):
        """
        This method locally trains the model with the dataset of the client. It handles the training at mini-batch level
        :param cur_epoch: current epoch of training
        :param optimizer: optimizer used for the local training
        """
        for cur_step, (images, labels) in enumerate(self.train_loader):
            # Get data to cuda if possible
            images = images.to(self.device)
            labels = labels.to(self.device)
        
            # forward
            outputs = self.model(images)

            loss = self.criterion(outputs, labels)

            # backward
            optimizer.zero_grad()
            loss.backward()

            #Clip norm
            #torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip)

            # gradient descent or adam step
            optimizer.step()



    def train(self, args):
        """
        This method locally trains the model with the dataset of the client. It handles the training at epochs level
        (by calling the run_epoch method for each local epoch of training)
        :return: length of the local dataset, copy of the model parameters
        """
        params = self.model.parameters()
        optmz = optim.SGD(params=params, lr=args.lr, momentum=args.m, weight_decay=args.wd)

        for epoch in range(self.args.num_epochs):
            self.run_epoch(epoch, optimizer=optmz)

        new_sd = self.model.state_dict()
        del self.model

        ## model.parameters returns
        return len(self.train_loader.dataset), new_sd
    
    def test(self, metric):
        """
        This method tests the model on the local dataset of the client.
        :param metric: StreamMetric object
        """
        with torch.no_grad():
            for i, (img, labels) in enumerate(self.test_loader):
                img = img.to(device=self.device)
                labels = labels.to(device=self.device)
                outputs = self._get_outputs(img)
                self.update_metric(metric, outputs, labels)

    def change_model(self, model, dcopy=True):
        #To deepcopy the model for doing the local training phase
        self.model = copy.deepcopy(model) if dcopy else model

    def get_local_loss(self):
        local_loss = 0
        for cur_step, (images, labels) in enumerate(self.train_loader):

            images = images.to(self.device)
            labels = labels.to(self.device)

            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            local_loss += loss.item() * len(labels)

        local_loss = local_loss / len(self.dataset)

        return local_loss