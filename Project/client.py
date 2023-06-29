import copy
import torch
import numpy as np
from torch import optim, nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import copy

from utils.utils import HardNegativeMining, MeanReduction


class Client:

    def __init__(self, args, dataset, model, test_client=False, device=None):
        self.args = args
        self.dataset = dataset
        self.name = self.dataset.client_name
        self.model = model
        self.r = 0

        self.train_loader = DataLoader(self.dataset, batch_size=self.args.bs, shuffle=True, drop_last=False) \
            if not test_client else None
        self.test_loader = DataLoader(self.dataset, batch_size=self.args.bs, shuffle=False)

        self.criterion = nn.CrossEntropyLoss(ignore_index=255)

        self.device = device

        if self.args.model == "dann":
            self.domain_criterion = nn.CrossEntropyLoss()
        if self.args.model == "fedsr":
            self.r_mu = nn.Parameter(torch.zeros(args.num_classes, args.z_dim).to(self.device))
            self.r_sigma = nn.Parameter(torch.ones(args.num_classes, args.z_dim).to(self.device))
            self.C = nn.Parameter(torch.ones([]).to(self.device))


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
        elif self.args.model == 'cnn' or self.args.model == 'fedsr':
            return self.model(images)
        elif self.args.model == 'dann':
            return self.model(images)[0]
        raise NotImplementedError

    def run_epoch(self, cur_epoch, optimizer):
        """
        This method locally trains the model with the dataset of the client. It handles the training at mini-batch level
        :param cur_epoch: current epoch of training
        :param optimizer: optimizer used for the local training
        """

        self.cumulative_cls_loss = 0
        self.cumulative_dmn_loss = 0

        for cur_step, (images, labels) in enumerate(self.train_loader):
            images = images.to(self.device)
            labels = labels.to(self.device)
        
            # forward
            if self.args.model == "fedsr":
                z, (z_mu, z_sigma) = self.model.featurise(images, return_dist=True)
                outputs = self.model.cls(z)

                loss = self.criterion(outputs, labels)

                if self.args.l2r != 0.0: 
                    regL2R = z.norm(dim=1).mean()
                    loss = loss + self.args.l2r * regL2R
                if self.args.cmi != 0.0:
                    r_sigma_softplus = F.softplus(self.r_sigma)
                    r_mu = self.r_mu[labels]
                    r_sigma = r_sigma_softplus[labels]
                    z_mu_scaled = z_mu * self.C
                    z_sigma_scaled = z_sigma * self.C
                    regCMI = torch.log(r_sigma) - torch.log(z_sigma_scaled) + \
                             (z_sigma_scaled ** 2 + (z_mu_scaled - r_mu) ** 2) / (2 * r_sigma ** 2) - 0.5
                    regCMI = regCMI.sum(1).mean()
                    loss = loss + self.args.cmi * regCMI

            ### FORWARD PROCEDURE FOR DANN
            elif self.args.model == "dann":
                domain_labels = self.dataset.domain * torch.ones(len(labels), dtype=labels.dtype)
                domain_labels = domain_labels.to(self.device)
                # n_domains=6

                # domain_labels_soft = torch.zeros((len(labels), n_domains), dtype=torch.float)
                # for i in range(len(domain_labels_soft)):
                #     domain_labels_soft[i, int(domain_labels[i])] = 1.0
                # domain_labels_soft = domain_labels_soft.to(self.device)

                outputs, domain_output = self.model(images)

                if self.args.dann_decay:
                    gamma = 10  
                    p = self.r / self.args.num_rounds
                    lambda_p = (2 / (1 + np.exp(-gamma * p))) - 1
                else:
                    lambda_p = self.args.dann_w


                loss_label = self.criterion(outputs, labels)
                loss_domain = self.domain_criterion(domain_output, domain_labels)

                self.cumulative_cls_loss += loss_label.item()
                self.cumulative_dmn_loss += loss_domain.item()

                # loss = loss_domain
                loss = loss_label + lambda_p * loss_domain

            else:
               outputs = self.model(images)
               loss = self.criterion(outputs, labels)

            #outputs = self.model(images)


            # backward
            optimizer.zero_grad()
            loss.backward()

            # gradient descent or adam step
            optimizer.step()



    def train(self, args):
        """
        This method locally trains the model with the dataset of the client. It handles the training at epochs level
        (by calling the run_epoch method for each local epoch of training)
        :return: length of the local dataset, copy of the model parameters
        """
        optimizer = optim.SGD(params=self.model.parameters(), lr=args.lr, momentum=args.m, weight_decay=args.wd)
        if args.model == "fedsr":
            optimizer.add_param_group({'params': [self.r_mu, self.r_sigma, self.C]})

        for epoch in range(self.args.num_epochs):
            self.run_epoch(epoch, optimizer=optimizer)

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

    def test_domains(self, metric):
        if self.args.model == "dann":
            with torch.no_grad():
                for i, (img, labels) in enumerate(self.test_loader):
                    img = img.to(device=self.device)
                    labels_domain = self.dataset.domain * torch.ones(len(labels), dtype=labels.dtype)

                    _, outputs_domain = self.model(img)


                    self.update_metric(metric, outputs_domain, labels_domain)
        else:
            raise Exception("No model test if no dann")

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

    def set_r(self, r):
        self.r = r


    def get_cls_loss(self):
        return self.cumulative_cls_loss/len(self.dataset)
    def get_dmn_loss(self):
        return self.cumulative_dmn_loss/len(self.dataset)