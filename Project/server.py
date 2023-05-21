import copy

import wandb

import torch.optim
import random
import numpy as np
import torch
import gc
import sys

import pandas as pd 



class Server:

    def __init__(self, args, train_clients, test_clients, model, metrics):
        self.args = args
        self.train_clients, self.validation_clients = self.split_train_val(train_clients)
        self.test_clients = test_clients
        self.model = model
        self.optim = torch.optim.SGD(params=model.parameters(), lr=1, momentum = self.args.sm)#, weight_decay=args.wd)
        self.metrics = metrics
        #self.model_params_dict = copy.deepcopy(self.model.state_dict())

    def count_labels(self, clients): 
        res = list()
        for element in clients: 
            res += [element.dataset.samples[i][1] for i in range(len(element.dataset.samples))]
        
        return pd.Series(res).value_counts().to_string()

    def split_train_val(self, train_clients): 
        random.shuffle(train_clients)
        l = len(train_clients)
        return train_clients[:int(l*self.args.tf)], train_clients[int(l*self.args.tf):]

    def select_clients(self):
        num_clients = min(self.args.clients_per_round, len(self.train_clients))
        return np.random.choice(self.train_clients, num_clients, replace=False)

    def biased_client_selection(self, prob, split):
        num_tot_clients = len(self.train_clients)
        client_split = int(split * num_tot_clients)

        picked_users = []

        while len(picked_users) < self.args.clients_per_round:
            rand_num = np.random.random()
            user = None

            while user == None:
                if rand_num <= prob:
                    # prendi high probability
                    user = np.random.choice(self.train_clients[ : client_split])
                else:
                    # prendi low probability
                    user = np.random.choice(self.train_clients[client_split : ])

                if user.name in [x.name for x in picked_users]:
                    user = None #Per ripetere iterazione nel caso in cui dovessi selezionare due volte lo stesso client

            picked_users.append(user)

        return picked_users

    def power_of_choice_selection(self):
        if self.args.d < self.args.clients_per_round:
            raise Exception("Power of choice: m < d < K not respected")

        # Create dataset-size probabilities
        client_dataset_sizes = np.array([len(c.dataset) for c in self.train_clients])
        total_samples = np.sum(client_dataset_sizes)
        client_probabilities = client_dataset_sizes / total_samples
        # try uniform ?
        # Try log the clients and see distributions


        # Get first d clients
        A_client_set = np.random.choice(self.train_clients, size=self.args.d, replace=False, p=client_probabilities)


        # Update models
        self.model.eval()
        for c in A_client_set:
            c.change_model(self.model, dcopy=False)

        # Get losses
        A_client_set_losses = np.array([c.get_local_loss() for c in A_client_set])

        # Choose highest loss clients
        active_clients_index = np.argsort(-A_client_set_losses)[ : self.args.clients_per_round]
        active_clients = A_client_set[active_clients_index]

        return active_clients

    def train_round(self, clients, n_round, args):
        """
            This method trains the model with the dataset of the clients. It handles the training at single round level
            :param clients: list of all the clients to train
            :return: model updates gathered from the clients, to be aggregated
        """
        updates = []
        n = len(clients)

        for i, c in enumerate(clients):
            # loading bar
            sys.stdout.write('\r')
            j = (i + 1) / n
            sys.stdout.write("Round %d: [%-20s] %d%%" % (n_round+1, '=' * int(20 * j), 100 * j))
            sys.stdout.flush()

            n_samples, model_parameters = c.train(args)

            updates.append( (n_samples, model_parameters) )

        return updates

    def aggregate_old(self, updates):
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

    def aggregate(self, updates):

        # updates = [(abs_weight, update), ...]
        n = sum([u[0] for u in updates])
        new_grad = [torch.zeros_like(p) for p in self.model.parameters()]
#        new_grad = {}
#        for key in self.model.state_dict():
#            if self.model.state_dict()[key].requires_grad:
#                new_grad[key] = 0*self.model.state_dict()[key]
        model_params = [p for p in self.model.parameters()]
        for weight, update in updates:
            update_params = [p for p in update]
            for i in range(len(new_grad)):
                new_grad[i] += (model_params[i] - update_params[i]) * (weight/n)

        return new_grad

    def step(self, new_grad):

        self.optim.zero_grad()
        for i, p in enumerate(self.model.parameters()):
            if p.requires_grad:
                p.grad = torch.Tensor(new_grad[i])
        self.optim.step()



    def train(self, args):
        """
        This method orchestrates the training the evals and tests at rounds level
        """

        for r in range(self.args.num_rounds):

            # Choose clients

            if self.args.client_selection == 'random': 
                clients = self.select_clients()
            elif self.args.client_selection == 'biased1':
                # 1 for 0.5 10% and 0.5 90% of the dataset
                clients = self.biased_client_selection(0.5, 0.1)
            elif self.args.client_selection == 'biased2':
                # 2 for 0.0001 30% and 0.9999 70% of the dataset 
                clients = self.biased_client_selection(0.0001, 0.3)
            elif self.args.client_selection == 'pow':
                clients = self.power_of_choice_selection()
            
            self.model.train()
            # Update parameters
            for c in clients:
                c.change_model(self.model) #with deepcopy

            updates = self.train_round(clients, r, args)

            new_grad = self.aggregate(updates)
            self.step(new_grad)
            sys.stdout.write("\n")

            #new_parameters = self.aggregate_old()
            #self.model.load_state_dict(new_parameters) ### UPDATE THE GLOBAL MODEL

            if (r+1) % self.args.gc == 0:
                print("Doing Gargage Collector in GPU")
                gc.collect()
                torch.cuda.empty_cache()
                print("Done!")

            if (r+1) % self.args.eval_interval == 0:
                #Test on Validation dataset every eval_interval number of rounds
                self.model.eval()
                for c in self.validation_clients:
                    c.change_model(self.model, dcopy=False)
                self.eval_train(r)

            if (r+1) % self.args.test_interval == 0:
                #Test on Test dataset every test_interval number of rounds
                self.model.eval()
                for c in self.test_clients:
                    c.change_model(self.model, dcopy=False)
                self.test(r)

    def eval_train(self, n_round):
        """
        This method handles the evaluation on the validation clients
        """
        self.metrics['eval_train'].reset()

        n = len(self.validation_clients)
        for i,c in enumerate(self.validation_clients):
            # loading bar
            sys.stdout.write('\r')
            j = (i + 1) / n
            sys.stdout.write("Evaluating validation clients: [%-20s] %d%%" % ( '=' * int(20 * j), 100 * j))
            sys.stdout.flush()

            c.test(self.metrics['eval_train'])

        #To load results obtained on wandb
        results = self.metrics['eval_train'].get_results()
        for k, v in results.items(): 
            if k != 'Class Acc': 
                name = k + '_train'
                wandb.log({name: v, "n_round": n_round})
        print(self.metrics['eval_train'])

    def test(self, n_round):
        """
            This method handles the test on the test clients
        """
        self.metrics['test'].reset()

        n = len(self.test_clients)
        for i,c in enumerate(self.test_clients):
            ### loading bar
            sys.stdout.write('\r')
            j = (i + 1) / n
            sys.stdout.write("Evaluating test clients: [%-20s] %d%%" % ( '=' * int(20 * j), 100 * j))
            sys.stdout.flush()
            ###

            c.test(self.metrics['test'])
        
        #To load results obtained on wandb
        results = self.metrics['test'].get_results()
        for k, v in results.items():
            if k != 'Class Acc': 
                name = k + '_test'
                wandb.log({name: v, "n_round": n_round})
        print(self.metrics['test'])
