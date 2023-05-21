import argparse


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--dataset', type=str, default = 'femnist', choices=['idda', 'femnist'], required=False, help='dataset name')
    parser.add_argument('--client_selection', type=str, default = 'random', choices=['random', 'biased1', 'biased2', 'pow'], required=False, help='client selection')
    parser.add_argument('--dataset_selection', type=str, default = 'default', choices=['default', 'rotated', 'L1O'], required=False, help='client selection')
    parser.add_argument('--leftout', type=int, default=0, choices=[0, 1, 2, 3, 4, 5], help='angle index left out in l1O')
    parser.add_argument('--niid', action='store_true', default=False,
                        help='Run the experiment with the non-IID partition (IID by default). Only on FEMNIST dataset.')
    parser.add_argument('--test_mode', action='store_true', default=False,
                        help='Enable or disable test mode.')
    parser.add_argument('--model', type=str, default = 'cnn', help='model name')
    parser.add_argument('--num_rounds', type=int, default = 1000, help='number of rounds')
    parser.add_argument('--num_epochs', type=int, default = 1, help='number of local epochs')
    parser.add_argument('--clients_per_round', type=int, default = 10, help='number of clients trained per round')
    parser.add_argument('--hnm', action='store_true', default=False, help='Use hard negative mining reduction or not')
    parser.add_argument('--lr', type=float, default=0.1, help='learning rate')
    parser.add_argument('--bs', type=int, default=64, help='batch size')
    parser.add_argument('--wd', type=float, default=1e-4, help='weight decay')
    parser.add_argument('--m', type=float, default=0.9, help='momentum')
    parser.add_argument('--sm', type=float, default=0, help='server_momentum')
    parser.add_argument('--d', type=int, default=20, help='pow_d')
    #parser.add_argument('--clip', type=float, default=0.5, help='clipping gradient')
    parser.add_argument('--tf', type=float, default=0.8, help='train_fraction')
    parser.add_argument('--gc', type=int, default=1001, help='after how many rounds call the garbage collector for cleaning GPU')
    parser.add_argument('--change_lr_interval', type=int, default=1000, help='after how many epoches multiply the lr by 0.1')
    parser.add_argument('--print_train_interval', type=int, default=50, help='client print train interval')
    parser.add_argument('--print_test_interval', type=int, default=1000, help='client print test interval')
    parser.add_argument('--eval_interval', type=int, default=50, help='eval interval')
    parser.add_argument('--test_interval', type=int, default=1000, help='test interval')
    return parser
