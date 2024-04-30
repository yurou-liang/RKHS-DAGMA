from lbfgsb_scipy import LBFGSBScipy
import torch
import torch.nn as nn
import numpy as np
import math
import matplotlib.pyplot as plt
import numpy.linalg as la


torch.set_default_dtype(torch.float64)


class NotearsRKHS(nn.Module):
    """n: number of samples, d: num variables"""
    def __init__(self, n, d, kernel):
        super(NotearsRKHS, self).__init__() # inherit  nn.Module
        self.d = d
        self.n = n
        self.kernel = kernel
        alpha = torch.zeros(d, n)
        self.alpha = nn.Parameter(alpha) 

        # initialize coefficients beta
        self.beta = nn.Parameter(torch.zeros(d, d, n))
        self.delta = torch.ones(d, d)
        self.delta.fill_diagonal_(0)
        self.I = torch.eye(self.d)
    
    
    def get_parameters(self): # [d, n]
       alpha = self.alpha
       beta = self.beta
       return alpha, beta
    
    #@timing_decorator
    @torch.no_grad()
    def gaussian_kernel_matrix_and_grad(self, x, gamma=1): 
      # x: [n, d]; K: [n, n]; grad_K1: [d, n, n, d]: grad_K1[j, i, l, k]: gradient of k(x^i, x^l) wrt x^i_{k} without jth coordinate; 
      # grad_K2: [n, n, d]: gradient of k(x^i, x^l) wrt x^l_{k}; mixed_grad: [d, n, n, d, d]: mixed_grad[j, i, l, a, b] gradient of k(x^i, x^l) wrt x^i_{a} and x^l_{b} without jth coordinate
      # Compute pairwise squared Euclidean distances using broadcasting
      diff = x.unsqueeze(1) - x.unsqueeze(0) # [n, n, d]
      sq_dist = torch.einsum('jk, ilk -> jil', self.delta, diff**2) # [d, n, n]

      # Compute the Gaussian kernel matrix
      K = torch.exp(-sq_dist / (gamma ** 2)) # [d, n, n] K[j, i, l] = k(x^i, x^l) without jth coordinate
      
      # Compute the gradient of K with respect to X
      grad_K1 = -2 / (gamma ** 2) * torch.einsum('jil, ila -> jila', K, diff) # [d, n, n, d] 
      identity_mask = torch.eye(self.d, dtype=torch.bool)
      broadcastable_mask = identity_mask.view(self.d, 1, 1, self.d)
      expanded_mask = broadcastable_mask.expand(-1, self.n, self.n, -1)
      grad_K1[expanded_mask] = 0.0
      grad_K2 = -grad_K1

      outer_products_diff = torch.einsum('ila, ilb->ilab', diff, diff)  # Outer product of the differences [n, n, d, d]
      mixed_grad = (-4 / gamma**4) * torch.einsum('jil, ilab -> jilab', K, outer_products_diff) # Apply the formula for a != b [d, n, n, d, d]

      # Diagonal elements (i == j) need an additional term
      K_expanded = torch.einsum('jil,ab->jilab', K, self.I) #[d, n, n, d, d]
      mixed_grad += 2/gamma**2 * K_expanded

      expanded_identity_mask1 = identity_mask.view(self.d, 1, 1, 1, self.d).expand(self.d, self.n, self.n, self.d, self.d)
      expanded_identity_mask2 = identity_mask.view(self.d, 1, 1, self.d, 1).expand(self.d, self.n, self.n, self.d, self.d)

      # # Zero out elements in A where the mask is True
      mixed_grad[expanded_identity_mask1] = 0
      mixed_grad[expanded_identity_mask2] = 0

      return K, grad_K1, grad_K2, mixed_grad
    
    @torch.no_grad()
    def matern3_2_kernel_matrix_and_grad(self, x, gamma=1): # K: [n, n]; grad_K: [n, n, d]: gradient of k(x^l, x^i) wrt x^l_{a}
       #x = torch.tensor(x, requires_grad=True)
       diff = x.unsqueeze(1) - x.unsqueeze(0) #[n, n, d]
       dist = torch.sqrt(torch.sum(diff ** 2, dim=2)) #[n, n]
       K = (1 + np.sqrt(3)*dist / gamma)*torch.exp(-np.sqrt(3)*dist / gamma) #[n, n]
       temp = -3/(gamma**2)*torch.exp(-np.sqrt(3)*dist/gamma) #[n, n]
       grad_K = diff*temp.unsqueeze(2)

       return K, grad_K
    
    @torch.no_grad()
    def matern5_2_kernel_matrix_and_grad(self, x, gamma=1): # [n, n, d]: gradient of k(x^i, x^l) wrt x^i_{k}
       diff = x.unsqueeze(1) - x.unsqueeze(0) #[n, n, d]
       dist = torch.sqrt(torch.sum(diff ** 2, dim=2)) #[n, n]
       K = (1 + np.sqrt(5)*dist / gamma + 5*(dist**2)/(3*(gamma**2)))*torch.exp(-np.sqrt(5)*dist / gamma)
       temp = (-5/(3*gamma**2) - 5*np.sqrt(5)*dist/(3*(gamma**3)))*torch.exp(-np.sqrt(5)*dist/gamma) #[n, n]
       grad_K = diff*temp.unsqueeze(2)

       return K, grad_K
    

    def forward(self, x: torch.tensor): #[n, d] -> [n, d], forward(x)_{i,j} = estimation of x_j at ith observation 
      """
      x: data matrix of shape [n, d] (np.array)
      forward(x)_{l,j} = estimation of x_j at lth observation
      """
      if self.kernel == "gaussian":
        K = self.gaussian_kernel_matrix_and_grad(x)[0]
        #grad_K1 = self.gaussian_kernel_matrix_and_grad(x)[1]
        grad_K2 = self.gaussian_kernel_matrix_and_grad(x)[2]
      elif self.kernel == "matern3_2":
        K = self.matern3_2_kernel_matrix_and_grad(x)[0]
        grad_K2 = self.matern3_2_kernel_matrix_and_grad(x)[2]
      elif self.kernel == "matern5_2":
        K = self.matern5_2_kernel_matrix_and_grad(x)[0]
        grad_K2 = self.matern5_2_kernel_matrix_and_grad(x)[2]
      else:
        print("Given kernel is invalid.")
      beta = self.get_parameters()[1]  
      output1 = torch.einsum('jl, jil -> ij', self.alpha, K) # [n, d]
      output2 = torch.einsum('jal, jila -> ijl', beta, grad_K2) # [n, d, n]
      output2 = torch.sum(output2, dim = 2) # [n, d]
      output = output1 + output2 # [n, d]
      return output


    def fc1_to_adj(self, x: torch.tensor) -> torch.Tensor: # [d, d]
      if self.kernel == "gaussian":
        grad_K1 = self.gaussian_kernel_matrix_and_grad(x)[1] # [n, n, d]
        mixed_grad = self.gaussian_kernel_matrix_and_grad(x)[3] # [n, n, d, d]
      elif self.kernel == "matern3_2":
        grad_K1 = self.matern3_2_kernel_matrix_and_grad(x)[1]
      elif self.kernel == "matern5_2":
        grad_K1 = self.matern5_2_kernel_matrix_and_grad(x)[1]
      else:
        print("Given kernel is invalid.")

      weight1 = torch.einsum('jl, jilk -> kij', self.alpha, grad_K1) # [d, n, d]
      beta = self.get_parameters()[1]  
      weight2 = torch.einsum('jal, jilka -> kij', beta, mixed_grad) # [d, n, d]
      weight = weight1 + weight2
      weight = torch.sum(weight ** 2, dim = 1)/self.n # [d, d]

      return weight
    
    #expoential h
    # def h_func(self, x: torch.tensor):
    #   weight = self.fc1_to_adj(x)
    #   h = trace_expm(weight)-self.d
    #   return h
    
    # log determinant h
    def h_func(self, x: torch.tensor, t = 200):
      weight = self.fc1_to_adj(x)
      A = t*self.I - weight
      sign, logabsdet = torch.linalg.slogdet(A)
      h = -logabsdet + self.d * np.log(t)
      return h

    #spetrum h
    # def h_func(self, x: torch.tensor):
    #   weight = self.fc1_to_adj(x)
    #   w = torch.ones(self.d)
    #   for _ in range(10):
    #       w = weight @ w
    #       w = w / (torch.norm(w) + 1e-8)
    #   return w @ weight @ w 
    

    def mse(self, x: torch.tensor): # [1, 1]
      """compute the regularized iempirical L_risk of squared loss function, penalty: penalty for H_norm"""
      x_est = self.forward(x) # [n, d]
      squared_loss = 0.5 / self.n * torch.sum((x_est - x) ** 2)
      return squared_loss
    
    def complexity_reg(self, x: torch.tensor, lambda1, tau):
      if self.kernel == "gaussian":
          K = self.gaussian_kernel_matrix_and_grad(x)[0] # [n, n]
          mixed_grad = self.gaussian_kernel_matrix_and_grad(x)[3] # [n, n, d, d]
          K_grad2 = self.gaussian_kernel_matrix_and_grad(x)[2] # [n, n, d]
      elif self.kernel == "matern3_2":
          K = self.matern3_2_kernel_matrix_and_grad(x)[0] # [n, n]
      elif self.kernel == "matern5_2":
          K = self.matern5_2_kernel_matrix_and_grad(x)[0] # [n, n]
      else:
          print("Given kernel is invalid.") 
      beta = self.get_parameters()[1]  
      temp1 = torch.einsum('ji, jil -> jl', self.alpha, K) #[d, n]
      temp1 = (self.alpha*temp1).sum() 
      temp2 = torch.einsum('jal, jila -> ji', beta, K_grad2) #[d, n]
      temp2 = (self.alpha * temp2).sum()
      temp3 = torch.einsum('jbl, jilab -> jai', beta, mixed_grad) #[d, d, n]
      temp3 = (beta * temp3).sum()
      regularized = lambda1*tau*(temp1 + temp2 + temp3)
      return regularized
    
    def sparsity_reg(self, x: torch.tensor, tau):
      W = self.fc1_to_adj(x)
      help = torch.tensor(1e-8) # numerical stability
      W_sqrt = torch.sqrt(W+help)
      sparsity = torch.sum(W_sqrt)
      return 2*tau*sparsity
    

    
def dual_ascent_step(model, X, lambda1, tau, rho, mu, h, rho_max):
    x_torch = torch.from_numpy(X)
    h_new = None
    optimizer = LBFGSBScipy(model.parameters())

    while rho < rho_max:
        print("rho: ", rho)
        def closure():
            optimizer.zero_grad()
            h_val = model.h_func(x_torch)
            print("h_val: ", h_val)
            penalty = 0.5 * rho * h_val * h_val + mu * h_val
            squared_loss = model.mse(x_torch)
            complexity_reg = model.complexity_reg(x_torch, lambda1, tau)
            sparsity_reg = model.sparsity_reg(x_torch, tau) 
            obj = squared_loss + complexity_reg + sparsity_reg + penalty
            print('squared loss:', squared_loss.item())
            print('obj:', obj.item())
            obj.backward()
            return obj

        optimizer.step(closure)
        h_new = model.h_func(x_torch).item()
        print("h_new: ", h_new)
        if h_new > 0.25 * h:
                rho *= 10
        else:
            break

    mu += rho * h_new
        

    return rho, mu, h_new

def RKHS_nonlinear(model: nn.Module,
                    X: np.array,
                    lambda1: float = 0.,
                    tau: float = 0.,
                    mu: float = 0.,
                    max_iter: int = 100,
                    h_tol: float = 1e-8,
                    rho_max: float = 1e+16,
                    w_threshold: float = 0.1):
    rho, mu, h = 1.0, 0.0, np.inf
    for k in range(max_iter):
        rho, mu, h = dual_ascent_step(model, X, lambda1, tau, rho, mu, h, rho_max)
        print("iteration: ", k+1)
        if h <= h_tol or rho >= rho_max:
            break
    x_torch = torch.tensor(X, dtype=torch.float64, requires_grad=True)
    W_est = model.fc1_to_adj(x_torch)
    W_est = torch.sqrt(W_est)
    W_est = W_est.detach().numpy()
    #print(W_est)
    W_est[np.abs(W_est) < w_threshold] = 0
    #W_est[np.abs(W_est) >= w_threshold] = 1
    output = model.forward(x_torch)
    return W_est, output