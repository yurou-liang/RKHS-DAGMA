#from notears import nonlinear
from notears import NotearsSobolev, notears_nonlinear
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
        df = pd.read_csv(content_as_file, sep=' ', header=None, names=['X', 'Y'])
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
        X = result_array
        x = result_array[:, 0]
        y = result_array[:, 1]
        print(X.shape)
    else:
        raise ValueError(f'Failed to retrieve the file: Status code {response.status_code}')
    
    torch.set_default_dtype(torch.float64)
    d = X.shape[1]
    model = NotearsSobolev(d = d, k = 10)
    W_est_no_thresh, output = notears_nonlinear(model, X, lambda1=lambda1, lambda2=lambda2, w_threshold=0.0)
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
    filename = f'{index}th_data_NOTEARS_SOB_results.txt'
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
    plt.savefig(f'{index}_NOTEARS_SOB.png')
    plt.clf()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='Tuebingen Testing by NOTEARS_SOB',)
    current_directory = os.getcwd()

    # Print the current working directory
    print("Current working directory:", current_directory)
    os.chdir('./Tuebingen')
    causality_df = pd.read_csv('causality_df.csv')
    index_list = causality_df['index'].tolist()

    parser.add_argument('-i', '--index', nargs='+', default = index_list, type=int)
    parser.add_argument('-l1', '--lambda1', default=3e-2, type=float)
    parser.add_argument('-l2', '--lambda2', default=0.0, type=float)
    parser.add_argument('-thresh', default=0.3, type=float)
    args = parser.parse_args()

    for index in index_list:
        print("Index: ", index)
        try:
            NOTEARS_tuebingen_causality(index, args.lambda1, args.lambda2, args.thresh)
        except Exception as e:
            print(f"An error occurred with index {index}: {e}")