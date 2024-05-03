from ..notears import nonlinear
import torch
import time
import numpy as np
import matplotlib.pyplot as plt
import requests
import pandas as pd
import json
from sklearn.preprocessing import StandardScaler
from io import StringIO
import argparse
import os


def NOTEARS_tuebingen_causality(index, lambda1, lambda2, thresh):
    result = {}
    # load data
    url = 'https://webdav.tuebingen.mpg.de/cause-effect/pair' + str(index).zfill(4) + '.txt'
    response = requests.get(url)
    
    os.chdir('./Final/Tuebingen')
    causality_df = pd.read_csv('causality_df.csv')
    index_list = causality_df['index'].tolist()

    if index not in index_list:
        raise ValueError("invalid index.")

    # Check if the request was successful
    if response.status_code == 200:
        # Read the content of the file
        content = response.text
        
        # Turn the string content into a file-like object
        content_as_file = StringIO(content)
        
        # Read into a DataFrame assuming the delimiter is a tab. Adjust if necessary.
        df = pd.read_csv(content_as_file, sep=' ', header=None, names=['X', 'Y'])
        scaler = StandardScaler()
        df['X'] = scaler.fit_transform(df[['X']])
        df['Y'] = scaler.fit_transform(df[['Y']])
        # Convert the DataFrame into a NumPy array
        X = df.to_numpy()
    else:
        raise ValueError(f'Failed to retrieve the file: Status code {response.status_code}')
    
    torch.set_default_dtype(torch.float64)
    d = X.shape[1]
    model = nonlinear.NotearsMLP(dims=[d, 10, 1], bias=True)
    W_est_no_thresh, output = nonlinear.notears_nonlinear(model, X, lambda1=lambda1, lambda2=lambda2, w_threshold=0.0)
    W_est = (abs(W_est_no_thresh) > thresh)
    causality = causality_df[causality_df['index'] == index]["causality"].item()
    if causality == 0:
        if W_est[0, 1] == 1 and W_est[1, 0] == 0:
                estimation = 'correct'
        else:
                estimation = 'incorrect'
    else:
        if W_est[0, 1] == 0 and W_est[1, 0] == 1:
                estimation = 'correct'
        else:
                estimation = 'incorrect' 
    result = {'W_est_no_thresh': W_est_no_thresh.tolist(), 'estimation': estimation}
    filename = f'{index}th_data_NOTEARS_MLP_results.txt'
    with open(filename, 'w') as file:
        json.dump(result, file, indent=4)

    y_hat = output[:, 1].cpu().detach().numpy()
    plt.figure(figsize=(10, 6))  # Optional: specifies the figure size
    plt.scatter(df.iloc[:, 0], df.iloc[:, 1], label='y', color='blue', marker='o')  # Plot x vs. y1
    plt.scatter(df.iloc[:, 0], y_hat, label='y_est', color='red', marker='s') 
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    # save the plot
    plt.savefig(f'{index}_NOTEARS_MLP.png')
    plt.clf()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='Tuebingen Testing by NOTEARS_MLP',)

    parser.add_argument('-i', '--index', type=int)
    parser.add_argument('-l1', '--lambda1', default=2e-2, type=float)
    parser.add_argument('-l2', '--lambda2', default=0.0, type=float)
    parser.add_argument('-thresh', default=0.3, type=float)
    args = parser.parse_args()

    NOTEARS_tuebingen_causality(args.index, args.lambda1, args.lambda2, args.thresh)