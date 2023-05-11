import copy
from collections import OrderedDict

import wandb

import numpy as np
import torch
import gc
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

    def biased_client_selection(self):
        num_tot_clients = len(self.train_clients)
        high_prob = int(0.1 * num_tot_clients)
        low_prob = int(0.3 * num_tot_clients)
        other_prob = num_tot_clients - high_prob - low_prob

        picked_users = []
        while len(picked_users) < self.args.clients_per_round:
            rand_num = np.random.random()
            user = None

            while user == None:
                if rand_num <= 0.5:
                    # prendi high probability
                    user = np.random.choice(self.train_clients[ : high_prob])
                elif 0.5 < rand_num < 0.999:
                    # prendi mid probability
                    user = np.random.choice(self.train_clients[high_prob : num_tot_clients-low_prob])
                else:
                    # prendi low probability
                    user = np.random.choice(self.train_clients[num_tot_clients-low_prob : ])

                if user.name in [x.name for x in picked_users]:
                    user = None

            picked_users.append(user)

        return picked_users

    def power_of_choice_selection(self, d):
        # Create dataset-size probabilities
        client_dataset_sizes = np.array([len(c.dataset) for c in self.train_clients])
        total_samples = np.sum(client_dataset_sizes)
        client_probabilities = client_dataset_sizes / total_samples

        # Get first d clients
        # size = d*self.args.clients_per_round ?????
        A_client_set = np.random.choice(self.train_clients, size=d, replace=False, p=client_probabilities)

        # Update models
        for c in A_client_set:
            c.change_model(self.model, dcopy=False)

        # Get losses
        A_client_set_losses = np.array([c.get_local_loss() for c in A_client_set])

        # Choose highest loss clients
        active_clients_index = np.argmax(A_client_set_losses)[ : min(d,self.args.clients_per_round)]
        active_clients = A_client_set[active_clients_index]

        return active_clients

    def train_round(self, clients, n_round, args):
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

            #print("ANDIAMO SIAMO IN ", len(clients))
            #print('\n')
            #print('client n. ', i)
            #print(list(c.model.state_dict().items())[0])
            n_samples, model_parameters = c.train(args)
            #print(list(c.model.state_dict().items())[0])
            #input("press enter.")

            #print(n_samples, " ANDIAMO AL PROSSIMO CLIENT")
            updates.append( (n_samples, model_parameters) )
        #gc.collect()
        #torch.cuda.empty_cache()
        return updates

    def aggregate(self, updates):
        """
        This method handles the FedAvg aggregation
        :param updates: updates received from the clients
        :return: aggregated parameters
        """

        ### INITIALIZE NEW (ZERO) STATE DICTIONARY
        #print("I'm aggregating!")
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



    def train(self, args):
        """
        This method orchestrates the training the evals and tests at rounds level
        """

        for r in range(self.args.num_rounds):

            # Choose clients
            # client_index = self.select_clients()
            # clients = [self.train_clients[i] for i in client_index]

            clients = self.select_clients()
            
            # Update parameters
            for c in clients:
                c.change_model(self.model)

            updates = self.train_round(clients, r, args)
            new_parameters = self.aggregate(updates)
            sys.stdout.write("\n")

            #print(list(self.model.state_dict().items())[0])
            self.model.load_state_dict(new_parameters) ### UPDATE THE GLOBAL MODEL
            #print(list(self.model.state_dict().items())[0])
            #input("press enter.")

            if (r+1) % self.args.eval_interval == 0:
                gc.collect()
                torch.cuda.empty_cache()
                for c in self.train_clients:
                    c.change_model(self.model, dcopy=False)
                self.eval_train()
                #input("press enter.")

            if (r+1) % self.args.test_interval == 0:
                for c in self.test_clients:
                    c.change_model(self.model, dcopy=False)
                self.test()
                #input("press enter.")

    def eval_train(self):
        """
        This method handles the evaluation on the train clients
        """

        self.metrics['eval_train'].reset()

        for c in self.train_clients:
            c.test(self.metrics['eval_train'])

        results = self.metrics['eval_train'].get_results()
        for k, v in results.items(): 
            if k != 'Class Acc': 
                name = k + '_train'
                wandb.log({name: v})
        print(self.metrics['eval_train'])


    def test(self):
        """
            This method handles the test on the test clients
        """
        self.metrics['test'].reset()

        for c in self.test_clients:
            c.test(self.metrics['test'])

        results = self.metrics['test'].get_results()
        for k, v in results.items():
            if k != 'Class Acc': 
                name = k + '_test'
                wandb.log({name: v})
        print(self.metrics['test'])
