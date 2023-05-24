import copy
import torch

from torch import optim, nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
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
        self.test_loader = DataLoader(self.dataset, batch_size=self.args.bs, shuffle=False)

        self.criterion = nn.CrossEntropyLoss(ignore_index=255)
        self.reduction = HardNegativeMining() if self.args.hnm else MeanReduction()
        self.optim = optim.SGD(params=self.model.parameters(), lr=args.lr, momentum=args.m, weight_decay=args.wd)

        self.device = device
        
        self.r_mu = nn.Parameter(torch.zeros(62, args.z_dim)).to(self.device)
        self.r_sigma = nn.Parameter(torch.ones(62, args.z_dim)).to(self.device)
        self.C = nn.Parameter(torch.ones([])).to(self.device)
        # self.optim.add_param_group({'params':[self.r_mu, self.r_sigma, self.C]})

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
        elif self.args.model == 'fedsr': 
            return self.model(images)
        raise NotImplementedError

    def run_epoch(self, cur_epoch):
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
            if self.model == 'fedsr':
                z, (z_mu, z_sigma) = self.model.featurise(images, return_dist=True)
                outputs = self.model.cls(z)
            
            outputs = self.model(images)

            loss = self.criterion(outputs, labels)
            
            if self.args.l2r != 0.0: #0.01 works quite well (as starting point)
                regL2R = z.norm(dim=1).mean()
                loss = loss + self.args.l2r*regL2R
            if self.args.cmi != 0.0:
                r_sigma_softplus = F.softplus(self.r_sigma)
                r_mu = self.r_mu[labels]
                r_sigma = r_sigma_softplus[labels]
                z_mu_scaled = z_mu*self.C
                z_sigma_scaled = z_sigma*self.C
                regCMI = torch.log(r_sigma) - torch.log(z_sigma_scaled) + \
                        (z_sigma_scaled**2+(z_mu_scaled-r_mu)**2)/(2*r_sigma**2) - 0.5
                regCMI = regCMI.sum(1).mean()
                loss = loss + self.args.cmi*regCMI
            
            # backward
            self.optim.zero_grad()
            loss.backward()

            #Clip norm
            #torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip)

            # gradient descent or adam step
            self.optim.step()



    def train(self, args):
        """
        This method locally trains the model with the dataset of the client. It handles the training at epochs level
        (by calling the run_epoch method for each local epoch of training)
        :return: length of the local dataset, copy of the model parameters
        """
        params = self.model.parameters()
        # optmz = optim.SGD(params=params, lr=args.lr, momentum=args.m, weight_decay=args.wd)
        # optmz.add_param_group({'params':[self.r_mu,self.r_sigma,self.C],'lr':args.lr,'momentum':0.9})

        for epoch in range(self.args.num_epochs):
            self.run_epoch(epoch)

        new_sd = self.model.parameters()
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