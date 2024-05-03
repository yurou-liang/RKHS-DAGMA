import utils, RKHS_DAGMA_extractj
import torch
import numpy as np
import json
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='RKHADagma Testing',)

    parser.add_argument('-d', '--num_nodes', nargs='+', default=[10, 20, 30, 40], type=int)
    parser.add_argument('-g', '--ER_order', default=4, type=int)
    parser.add_argument('-N', '--num_trials', default=10, type=int)
    parser.add_argument('-T', '--num_iterations', default=6, type=int)
    parser.add_argument('-s', '--random_seed', default=0, type=int)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('-f', '--function_type',  nargs='+', default=['gp-add', 'gp', 'mlp'])
    parser.add_argument('-A', '--algorithm', default='RKHS', type=str)

    args = parser.parse_args()

    device = torch.device(args.device)
    torch.set_default_device(device)

    torch.set_default_dtype(torch.double)
    utils.set_random_seed(args.random_seed)
    torch.manual_seed(args.random_seed)


    for idx_nodes, n_nodes in enumerate(args.num_nodes):
        print('-----------------------------\n' +
              f'| Experiments with {n_nodes} Nodes |\n' +
              '-----------------------------\n')
        for t in range(args.num_trials):
            print(f'Trial {t+1}')