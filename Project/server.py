import copy
from collections import OrderedDict

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

    def train_round(self, clients):
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
            sys.stdout.write("Round %d: [%-20s] %d%%" % (i, '=' * int(20 * j), 100 * j))
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
        # TODO: missing code here!
        raise NotImplementedError

    def train(self):
        """
        This method orchestrates the training the evals and tests at rounds level
        """

        for r in range(self.args.num_rounds):

            # Chose clients
            client_index = self.select_clients()
            clients = [self.train_clients[i] for i in client_index]

            updates = self.train_round(clients)
            new_parameters = self.aggregate(updates)

            ######## !!! MA IL TEST/VALIDATION ?

            # Update parameters
            for c in self.train_clients:
                c.change_model(self.model)

    def eval_train(self):
        """
        This method handles the evaluation on the train clients
        """
        # TODO: missing code here!
        raise NotImplementedError

    def test(self):
        """
            This method handles the test on the test clients
        """
        # TODO: missing code here!
        raise NotImplementedError
