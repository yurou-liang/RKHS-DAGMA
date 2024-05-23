from RKHS import RKHS_DAGMA_extractj
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


def RKHS_tuebingen_causality(index, lambda1, tau, gamma, T, lr):
    result = {}
    # load data
    url = 'https://webdav.tuebingen.mpg.de/cause-effect/pair' + str(index).zfill(4) + '.txt'
    response = requests.get(url)

    #os.chdir('./Tuebingen')
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
        df = pd.read_csv(content_as_file, sep=r'\s+', header=None, names=['X', 'Y'])
        print(df.head())
        scaler = StandardScaler()
        df['X'] = scaler.fit_transform(df[['X']])
        df['Y'] = scaler.fit_transform(df[['Y']])
        # Convert the DataFrame into a NumPy array
        data_array = df.to_numpy()
        if len(data_array) > 400:
            data_array_sorted = data_array[data_array[:, 0].argsort()]
            num_points = len(data_array_sorted) // 300

            # Initialize the result array
            result_array = np.zeros((300, 2))

            # Process each grid
            for i in range(300):
                # Start and end indices of the grid
                start = i * num_points
                end = start + num_points if i < 299 else len(data_array_sorted)  # Adjust last grid to include remainder
                
                # Calculate the median of x1 in this grid
                median_x1 = np.median(data_array_sorted[start:end, 0])
                # We'll take the x2 value corresponding to the median x1 index
                median_index = start + (end - start) // 2
                corresponding_x2 = data_array_sorted[median_index, 1]
                
                # Store in result
                result_array[i, 0] = median_x1
                result_array[i, 1] = corresponding_x2
        else:
            result_array = data_array
        # Now 'array' is a NumPy array with the data from the text file
        X = torch.tensor(result_array)
        x = result_array[:, 0]
        y = result_array[:, 1]
        print(X.shape)
    else:
        raise ValueError(f'Failed to retrieve the file: Status code {response.status_code}')

    torch.set_default_dtype(torch.float64)
    device= torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.set_default_device(device)
    
    X = X.to(device)
    eq_model = RKHS_DAGMA_extractj.DagmaRKHS(X, gamma = gamma).to(device)
    model = RKHS_DAGMA_extractj.DagmaRKHS_nonlinear(eq_model)
    W_est_no_thresh, output = model.fit(X, lambda1=lambda1, tau=tau, T = T, mu_init = 1.0, lr=lr, w_threshold=0.0)
    result['W_est_no_thresh'] = W_est_no_thresh.tolist()
    causality = causality_df[causality_df['index'] == index]["causality"].item()
    thresh_values = [0.005, 0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
    for thresh in thresh_values:
        W_est_dagma = (abs(W_est_no_thresh) > thresh)
        if W_est_dagma[0, 1] + W_est_dagma[1, 0] == 1:
            valid_DAG = 'yes'
        else:
            valid_DAG = 'no'
        if causality == 0:
            if W_est_dagma[0, 1] == 1 and W_est_dagma[1, 0] == 0:
                estimation = 'correct'
            else:
                estimation = 'incorrect'
        else:
            if W_est_dagma[0, 1] == 0 and W_est_dagma[1, 0] == 1:
                estimation = 'correct'
            else:
                estimation = 'incorrect' 
        key = f'threshold_{thresh}'
        result[key] = {'valid DAG': valid_DAG, 'estimation': estimation}
    filename = f'{index}th_data_RKHS_results.txt'
    with open(filename, 'w') as file:
        json.dump(result, file, indent=4)

    y_hat = output[:, 1].cpu().detach().numpy()
    plt.figure(figsize=(10, 6))  # Optional: specifies the figure size
    plt.scatter(x, y, label='y', color='blue', marker='o')  # Plot x vs. y1
    plt.scatter(x, y_hat, label='y_est', color='red', marker='s') 
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    # save the plot
    plt.savefig(f'{index}_RKHS.png')
    plt.clf()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='Tuebingen Testing by RKHS',)
    current_directory = os.getcwd()

    # Print the current working directory
    #print("Current working directory:", current_directory)
    os.chdir('./Tuebingen')
    causality_df = pd.read_csv('causality_df.csv')
    index_list = causality_df['index'].tolist()

    parser.add_argument('-i', '--index', nargs='+', default = index_list, type=int)
    parser.add_argument('-l', '--lambda1', default=1e-3, type=float)
    parser.add_argument('-t', '--tau', default=1e-4, type=float)
    parser.add_argument('-g', '--gamma', default=0.8, type=int)
    parser.add_argument('-T', default=6, type=int)
    parser.add_argument('-lr', default=0.03, type=float)
    args = parser.parse_args()

    for index in args.index:
        try:
            RKHS_tuebingen_causality(index, args.lambda1, args.tau, args.gamma, args.T, args.lr)
        except Exception as e:
            print(f"An error occurred with index {index}: {e}")
