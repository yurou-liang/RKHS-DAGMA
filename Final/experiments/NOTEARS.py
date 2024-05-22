import concurrent.futures
import sys
import os
import json
from experiments import utils
from RKHS import RKHS_DAGMA_extractj
import sys
from notears import NotearsMLP, notears_nonlinear, NotearsSobolev, notears_linear
import torch
import numpy as np
import json
import argparse
import itertools

current_directory = os.getcwd()
relative_path = 'experiments\\RKHS'
full_path = os.path.join(current_directory, relative_path)

def squared_loss(output, target):
    n = target.shape[0]
    loss = 0.5 / n * torch.sum((output - target) ** 2)
    return loss

### mlp ###
def NOTEARS_MLP(X, B_true, function_type, d, seed):
    results = {}
    model = NotearsMLP(dims=[d, 10, 1], bias=True)
    W_est, output = notears_nonlinear(model, X, lambda1=2e-2)
    acc = utils.count_accuracy(B_true, W_est != 0)
    diff = np.linalg.norm(W_est - abs(B_true))
    X_torch = torch.from_numpy(X)
    x_est = model.forward(X_torch)
    mse = squared_loss(x_est, X_torch)
    h_val = model.h_func().detach().cpu().numpy()
    filename = f'NOTEARS_MLP_function_type_{function_type}_d{d}_seed{seed}'
    results = {
    'SHD': acc['shd'],
    'TPR': acc['tpr'],
    'F1': acc['f1'],
    'diff': diff,
    'mse': mse.item(),
    'h_val': h_val.item(),
    'W_est': W_est.tolist(),
    'B_true': B_true.tolist(),
    'X': X.tolist()}

    with open(filename, 'w') as file:
        json.dump(results, file, indent=4) 

    
def process_NOTEARS_MLP(d, seed, function_type):
    filename = f'RKHS_function_type_{function_type}_d{d}_seed{seed}'
    file_path = os.path.join(full_path, filename)
    with open(file_path, 'r') as file:
        data = json.load(file)
        B_true = np.array(data['B_true'])
        X = np.array(data['X'])
    
    NOTEARS_MLP(X, B_true, function_type, d, seed)

def process_pair_NOTEARS_MLP(tuple):
    d, seed, function_type = tuple
    process_NOTEARS_MLP(d, seed, function_type)

### sob ###
def NOTEARS_SOB(X, B_true, function_type, d, seed):
    results = {}
    model = NotearsSobolev(d = d, k = 10)
    W_est, output = notears_nonlinear(model, X, lambda1=3e-2)
    acc = utils.count_accuracy(B_true, W_est != 0)
    diff = np.linalg.norm(W_est - abs(B_true))
    X_torch = torch.from_numpy(X)
    x_est = model.forward(X_torch)
    mse = squared_loss(x_est, X_torch)
    h_val = model.h_func().detach().cpu().numpy()
    filename = f'NOTEARS_SOB_function_type_{function_type}_d{d}_seed{seed}'
    results = {
    'SHD': acc['shd'],
    'TPR': acc['tpr'],
    'F1': acc['f1'],
    'diff': diff,
    'mse': mse.item(),
    'h_val': h_val.item(),
    'W_est': W_est.tolist(),
    'B_true': B_true.tolist(),
    'X': X.tolist()}

    with open(filename, 'w') as file:
        json.dump(results, file, indent=4) 

def process_NOTEARS_SOB(d, seed, function_type):
    filename = f'RKHS_function_type_{function_type}_d{d}_seed{seed}'
    file_path = os.path.join(full_path, filename)
    with open(file_path, 'r') as file:
        data = json.load(file)
        B_true = np.array(data['B_true'])
        X = np.array(data['X'])
    
    NOTEARS_SOB(X, B_true, function_type, d, seed)


def process_pair_NOTEARS_SOB(tuple):
    d, seed, function_type = tuple
    process_NOTEARS_SOB(d, seed, function_type)

### linear NOTEARS ###
def LINEAR_NOTEARS(X, B_true, function_type, d, seed):
    results = {}
    W_est = notears_linear(X, lambda1=0.1, loss_type = 'l2')
    acc = utils.count_accuracy(B_true, W_est != 0)
    diff = np.linalg.norm(W_est - abs(B_true))
    filename = f'LINEAR_NOTEARS_function_type_{function_type}_d{d}_seed{seed}'
    results = {
    'SHD': acc['shd'],
    'TPR': acc['tpr'],
    'F1': acc['f1'],
    'diff': diff,
    'W_est': W_est.tolist(),
    'B_true': B_true.tolist(),
    'X': X.tolist()}

    with open(filename, 'w') as file:
        json.dump(results, file, indent=4) 

def process_LINEAR_NOTEARS(d, seed, function_type):
    filename = f'RKHS_function_type_{function_type}_d{d}_seed{seed}'
    file_path = os.path.join(full_path, filename)
    with open(file_path, 'r') as file:
        data = json.load(file)
        B_true = np.array(data['B_true'])
        X = np.array(data['X'])
    
    LINEAR_NOTEARS(X, B_true, function_type, d, seed)

def process_pair_LINEAR_NOTEARS(tuple):
    d, seed, function_type = tuple
    process_LINEAR_NOTEARS(d, seed, function_type)



def main(d_value, seeds, function_type, algorithm):
    all_combinations = list(itertools.product(d_value, seeds, function_type))
    if algorithm == "NOTEARS_MLP":
        # Using ProcessPoolExecutor to process the function in parallel
        with concurrent.futures.ProcessPoolExecutor() as executor:
            # Use map to apply the square_number function to all items in numbers
            executor.map(process_pair_NOTEARS_MLP, all_combinations)
    elif algorithm == "NOTEARS_SOB":
        with concurrent.futures.ProcessPoolExecutor() as executor:
        # Use map to apply the square_number function to all items in numbers
            executor.map(process_pair_NOTEARS_SOB, all_combinations)
    elif algorithm == "LINEAR_NOTEARS":
        with concurrent.futures.ProcessPoolExecutor() as executor:
        # Use map to apply the square_number function to all items in numbers
            executor.map(process_pair_LINEAR_NOTEARS, all_combinations)
    else:
        print("Given algorithm is not valid.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='run NOTEARS algorithm',)

    parser.add_argument('-d', '--num_nodes', nargs='+', default=[10, 20, 30, 40], type=int)
    parser.add_argument('-f', '--function_type',  nargs='+', default=['gp-add', 'gp', 'mlp'], type=str)
    parser.add_argument('-s', '--random_seed', nargs='+', default=[0], type=int)
    parser.add_argument('-A', '--algorithm', default='NOTEARS_MLP', type=str)
    args = parser.parse_args()
    #print("d type: ", type(args.num_nodes))
    torch.set_default_dtype(torch.double)
    print('-----------------------------\n' +
        f'| Experiments with {args.num_nodes} Nodes |\n' +
        f'| function type {args.function_type} |\n' +
        f'| by algorithm {args.algorithm} |\n' +
        '-----------------------------\n')
    main(d_value = args.num_nodes, seeds = args.random_seed, function_type = args.function_type, algorithm = args.algorithm)