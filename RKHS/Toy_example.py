import torch
import numpy as np
import matplotlib.pyplot as plt
import RKHS_DAGMA
import argparse

np.random.seed(0)
torch.set_default_dtype(torch.float64)
device= torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_default_device(device)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='Toy example',)
    parser.add_argument('-r', '--relationship', default = 'quadratic', type=str)
    args = parser.parse_args()
    print('-----------------------------\n' +
        f'| Toy example with {args.relationship} relationship |\n' +
        '-----------------------------\n')
    epsilon = np.random.normal(0,1, 100)
    if args.relationship == "quadratic":
        x = np.random.uniform(low=0, high=10, size=100) 
        y = np.array([x**2 + epsilon for x, epsilon in zip(x, epsilon)])
    elif args.relationship == "inverse":
        x = np.random.uniform(low=0.001, high=1, size=100)
        y = np.array([1/x + epsilon for x, epsilon in zip(x, epsilon)])
    elif args.relationship == "qubic":
        x = np.random.uniform(low=-10, high=10, size=100)
        y = np.array([x**3 + x + epsilon for x, epsilon in zip(x, epsilon)])
    elif args.relationship == "sinus":
        x = np.random.uniform(low=-3, high=3, size=100)
        y = np.array([np.sin(x)*10 + epsilon for x, epsilon in zip(x, epsilon)])
    else:
        raise ValueError("Given realtionship is invalid.")

    X = np.column_stack((x, y))
    X = torch.from_numpy(X).to(device)
    eq_model = RKHS_DAGMA.RKHSDagma(X, gamma = 1).to(device)
    model = RKHS_DAGMA.RKHSDagma_nonlinear(eq_model)
    W_est_no_thresh, output = model.fit(X, lambda1=1e-3, tau=1e-4, T = 6, mu_init = 1.0, lr=0.03, w_threshold=0.0)
    B_true = np.array([[0, 1], [0, 0]])
    print("B_true: ", B_true)
    print("W_est_no_threshold: ", W_est_no_thresh)

    y_hat = output[:, 1].cpu().detach().numpy()
    plt.figure(figsize=(10, 6))  # Optional: specifies the figure size
    plt.scatter(x, y, label='y', color='blue', marker='o')  # Plot x vs. y1
    plt.scatter(x, y_hat, label='y_est', color='red', marker='s') 
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.savefig(f'Toy_example_{args.relationship}.png')
    plt.show()