import copy
from collections import OrderedDict

import wandb

import numpy as np
import torch
import sys


class Server:

    def __init__(self, args, train_clients, test_clients, model, metrics):
        self.args = args
        self.train_clients = train_clients
        self.test_clients = test_clients
        self.model = model
        self.metrics = metrics
        self.model_params_dict = copy.deepcopy(self.model.state_dict())

    def select_clients(self):
        num_clients = min(self.args.clients_per_round, len(self.train_clients))
        return np.random.choice(self.train_clients, num_clients, replace=False)

    def train_round(self, clients, n_round):
        """
            This method trains the model with the dataset of the clients. It handles the training at single round level
            :param clients: list of all the clients to train
            :return: model updates gathered from the clients, to be aggregated
        """
        updates = []
        for i, c in enumerate(clients):
            # loading bar
            n = len(clients)
            sys.stdout.write('\r')
            j = (i + 1) / n
            sys.stdout.write("Round %d: [%-20s] %d%%" % (n_round+1, '=' * int(20 * j), 100 * j))
            sys.stdout.flush()

            n_samples, model_parameters = c.train()
            updates.append( (n_samples, model_parameters) )
        return updates

    def aggregate(self, updates):
        """
        This method handles the FedAvg aggregation
        :param updates: updates received from the clients
        :return: aggregated parameters
        """

        ### INITIALIZE NEW (ZERO) STATE DICTIONARY
        model_sd = self.model.state_dict()
        new_sd = {}
        total_count = 0
        for key in model_sd:
            new_sd[key] = 0*model_sd[key]

        ### AVERAGE THE OTHER STATE DICTIONARIES
        for num, state_dict in updates:
            total_count += num
            for key in model_sd:
                new_sd[key] += num * state_dict[key]

        for key in new_sd:
            new_sd[key] = new_sd[key]/total_count

        return new_sd



    def train(self):
        """
        This method orchestrates the training the evals and tests at rounds level
        """

        for r in range(self.args.num_rounds):

            # Choose clients
            # client_index = self.select_clients()
            # clients = [self.train_clients[i] for i in client_index]

            clients = self.select_clients()
            
            updates = self.train_round(clients, r)
            new_parameters = self.aggregate(updates)
            sys.stdout.write("\n")

            self.model.load_state_dict(new_parameters) ### UPDATE THE GLOBAL MODEL

            # Update parameters
            for c in self.train_clients:
                c.change_model(self.model)

            if (r+1) % self.args.eval_interval == 0:
                self.eval_train()

            if (r+1) % self.args.test_interval == 0:
                self.test()

    def eval_train(self):
        """
        This method handles the evaluation on the train clients
        """

        self.metrics['eval_train'].reset()

        for c in self.train_clients:
            c.test(self.metrics['eval_train'])

        self.metrics['eval_train'].get_results()
        wandb.log({"metrics train": self.metrics['eval_train']})
        print(self.metrics['eval_train'])


    def test(self):
        """
            This method handles the test on the test clients
        """
        self.metrics['test'].reset()

        for c in self.test_clients:
            c.test(self.metrics['test'])

        self.metrics['test'].get_results()
        wandb.log({"metrics test": self.metrics['test']})
        print(self.metrics['test'])
