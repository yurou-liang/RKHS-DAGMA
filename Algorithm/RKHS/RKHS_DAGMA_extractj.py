import torch
import torch.nn as nn
import numpy as np
import math
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import typing
from  torch import optim
import copy


torch.set_default_dtype(torch.float64)
device= torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_default_device(device)


class DagmaRKHS(nn.Module):
    """n: number of samples, d: num variables"""
    def __init__(self, n, d, kernel):
        super(DagmaRKHS, self).__init__() # inherit  nn.Module
        self.d = d
        self.n = n
        self.kernel = kernel

        # initialize coefficients alpha
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
    
    @torch.no_grad()
    def gaussian_kernel_matrix_and_grad(self, x, gamma=1): 
      # x: [n, d]; K: [n, n]; grad_K1: [n, n, d]: gradient of k(x^i, x^l) wrt x^i_{k}; grad_K2: [n, n, d]: gradient of k(x^i, x^l) wrt x^l_{k}; mixed_grad: [n, n, d, d] gradient of k(x^i, x^l) wrt x^i_{a} and x^l_{b}
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
       #x = torch.tensor(x, requires_grad=True)
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
    def h_func(self, x: torch.tensor, s: torch.tensor):
      s = torch.tensor(s, dtype=torch.float64)
      weight = self.fc1_to_adj(x)
      #print("weight:", weight)
      A = s*self.I - weight
      sign, logabsdet = torch.linalg.slogdet(A)
      #print("t type: ", t.dtype)
      h = -logabsdet + self.d * torch.log(s)
      #print("h: ", h)
      return h

    #spetrum h
    # def h_func(self, x: torch.tensor):
    #   weight = self.fc1_to_adj(x)
    #   w = torch.ones(self.d)
    #   for _ in range(10):
    #       w = weight @ w
    #       w = w / (torch.norm(w) + 1e-8)
    #   return w @ weight @ w
    

    def L_risk(self, x: torch.tensor, lambda1, tau): # [1, 1]
      """compute the regularized iempirical L_risk of squared loss function, penalty: penalty for H_norm"""
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
      x_est = self.forward(x) # [n, d]
      squared_loss = 0.5 / self.n * torch.sum((x_est - x) ** 2)
      temp1 = torch.einsum('ji, jil -> jl', self.alpha, K) #[d, n]
      temp1 = (self.alpha*temp1).sum() 
      temp2 = torch.einsum('jal, jila -> ji', beta, K_grad2) #[d, n]
      temp2 = (self.alpha * temp2).sum()
      temp3 = torch.einsum('jbl, jilab -> jai', beta, mixed_grad) #[d, d, n]
      temp3 = (beta * temp3).sum()
      regularized = lambda1*(temp1 + temp2 + temp3)
      W = self.fc1_to_adj(x)
      help = torch.tensor(1e-8) # numerical stability
      W_sqrt = torch.sqrt(W+help)
      sparsity = torch.sum(W_sqrt)
      loss = squared_loss + tau*(regularized + 2*sparsity)
      return loss
    

class DagmaRKHS_nonlinear:
    """
    Class that implements the DAGMA algorithm
    """

    def __init__(self, model: nn.Module, verbose: bool = False, dtype: torch.dtype = torch.float64):
        """
        Parameters
        ----------
        model : nn.Module
            Neural net that models the structural equations.
        verbose : bool, optional
            If true, the loss/score and h values will print to stdout every ``checkpoint`` iterations,
            as defined in :py:meth:`~dagma.nonlinear.DagmaNonlinear.fit`. Defaults to ``False``.
        dtype : torch.dtype, optional
            float number precision, by default ``torch.double``.
        """
        self.vprint = print if verbose else lambda *a, **k: None
        #self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.model = model#.to(self.device)

        self.dtype = dtype

        
    def minimize(self, 
                max_iter: float, 
                lr: float, 
                lambda1: float, 
                tau: float,
                lambda2: float, 
                mu: float, 
                s: float,
                lr_decay: float = False, 
                tol: float = 1e-6, 
                pbar: typing.Optional[tqdm] = None,
        ) -> bool:
        r"""
        Solves the optimization problem: 
            .. math::
                \arg\min_{W(\Theta) \in \mathbb{W}^s} \mu \cdot Q(\Theta; \mathbf{X}) + h(W(\Theta)),
        where :math:`Q` is the score function, and :math:`W(\Theta)` is the induced weighted adjacency matrix
        from the model parameters. 
        This problem is solved via (sub)gradient descent using adam acceleration.

        Parameters
        ----------
        max_iter : float
            Maximum number of (sub)gradient iterations.
        lr : float
            Learning rate.
        lambda1 : float
            function penalty coefficient. 
        tau : float
            sparsity penalty coefficient.
        lambda2 : float
            L2 penalty coefficient. Applies to all the model parameters.
        mu : float
            Weights the score function.
        s : float
            Controls the domain of M-matrices.
        lr_decay : float, optional
            If ``True``, an exponential decay scheduling is used. By default ``False``.
        tol : float, optional
            Tolerance to admit convergence. Defaults to 1e-6.
        pbar : tqdm, optional
            Controls bar progress. Defaults to ``tqdm()``.

        Returns
        -------
        bool
            ``True`` if the optimization succeded. This can be ``False`` when at any iteration, the model's adjacency matrix 
            got outside of the domain of M-matrices.
        """
        self.vprint(f'\nMinimize s={s} -- lr={lr}')

        # for param in self.model.parameters():
        #     print('param.device',param.device, param.dtype)


        optimizer = optim.Adam(self.model.parameters(), lr=lr, betas=(.99,.999), weight_decay=mu*lambda2)
        if lr_decay is True:
            scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.8)
        obj_prev = 1e16
        for i in range(max_iter):
            optimizer.zero_grad()
            h_val = self.model.h_func(self.X, s)
            #print("h_val: ", h_val.device, h_val.dtype)
            if h_val.item() < 0:
                self.vprint(f'Found h negative {h_val.item()} at iter {i}')
                return False
            score = self.model.L_risk(self.X, lambda1, tau)
            print("squared loss: ", score)
            print("mu: ", mu)
            print("h_val: ", h_val)
            obj = mu * score + h_val
            print("obj: ", obj)
            obj.backward()
            optimizer.step()
            if lr_decay and (i+1) % 1000 == 0: #every 1000 iters reduce lr
                scheduler.step()
            if i % self.checkpoint == 0 or i == max_iter-1:
                obj_new = obj.item()
                self.vprint(f"\nInner iteration {i}")
                self.vprint(f'\th(W(model)): {h_val.item()}')
                self.vprint(f'\tscore(model): {obj_new}')
                if np.abs((obj_prev - obj_new) / obj_prev) <= tol:
                    pbar.update(max_iter-i)
                    break
                obj_prev = obj_new
            pbar.update(1)
        return True
    

    def fit(self, 
        X: torch.Tensor,
        lambda1: torch.float64 = .02, 
        tau: torch.float64 = .02,
        lambda2: torch.float64 = .005,
        T: torch.int = 4, 
        mu_init: torch.float64 = 0.1, 
        mu_factor: torch.float64 = .1, 
        s: torch.float64 = 1.0,
        warm_iter: torch.int = 5e3, 
        max_iter: torch.int = 8e3, 
        lr: torch.float64 = .005, 
        w_threshold: torch.float64 = 0.3, 
        checkpoint: torch.int = 1000,
    ) -> np.ndarray:
        r"""
        Runs the DAGMA algorithm and fits the model to the dataset.

        Parameters
        ----------
        X : typing.Union[torch.Tensor, np.ndarray]
            :math:`(n,d)` dataset.
        lambda1 : float, optional
            Coefficient of the function penalty, by default .02.
        lambda2 : float, optional
            Coefficient of the L2 penalty, by default .005.
        T : int, optional
            Number of DAGMA iterations, by default 4.
        mu_init : float, optional
            Initial value of :math:`\mu`, by default 0.1.
        mu_factor : float, optional
            Decay factor for :math:`\mu`, by default .1.
        s : float, optional
            Controls the domain of M-matrices, by default 1.0.
        warm_iter : int, optional
            Number of iterations for :py:meth:`~dagma.nonlinear.DagmaNonlinear.minimize` for :math:`t < T`, by default 5e3.
        max_iter : int, optional
            Number of iterations for :py:meth:`~dagma.nonlinear.DagmaNonlinear.minimize` for :math:`t = T`, by default 8e3.
        lr : float, optional
            Learning rate, by default .0002.
        w_threshold : float, optional
            Removes edges with weight value less than the given threshold, by default 0.3.
        checkpoint : int, optional
            If ``verbose`` is ``True``, then prints to stdout every ``checkpoint`` iterations, by default 1000.

        Returns
        -------
        np.ndarray
            Estimated DAG from data.
        
        
        .. important::

            If the output of :py:meth:`~dagma.nonlinear.DagmaNonlinear.fit` is not a DAG, then the user should try larger values of ``T`` (e.g., 6, 7, or 8) 
            before raising an issue in github.
        """
        torch.set_default_dtype(self.dtype)
        self.X = X
        self.checkpoint = checkpoint
        mu = mu_init
        if type(s) == list:
            if len(s) < T: 
                self.vprint(f"Length of s is {len(s)}, using last value in s for iteration t >= {len(s)}")
                s = s + (T - len(s)) * [s[-1]]
        elif type(s) in [int, float]:
            s = T * [s]
        else:
            ValueError("s should be a list, int, or float.") 

        with tqdm(total=(T-1)*warm_iter+max_iter) as pbar:
            for i in range(int(T)):
                self.vprint(f'\nDagma iter t={i+1} -- mu: {mu}', 30*'-')
                success, s_cur = False, s[i]
                inner_iter = int(max_iter) if i == T - 1 else int(warm_iter)
                model_copy = copy.deepcopy(self.model)
                lr_decay = False
                while success is False:
                    success = self.minimize(inner_iter, lr, lambda1, tau, lambda2, mu, s_cur, 
                                        lr_decay, pbar=pbar)
                    if success is False:
                        self.model.load_state_dict(model_copy.state_dict().copy()) # restore the model parameters to last iteration
                        # reset lr, lr_decay, s_cur then update the model
                        lr *= 0.5 
                        lr_decay = True
                        if lr < 1e-10:
                            print(":(")
                            break # lr is too small
                        s_cur = 1
                mu *= mu_factor
        W_est = self.model.fc1_to_adj(self.X)
        W_est = torch.sqrt(W_est)
        W_est = W_est.cpu().detach().numpy()
        print(W_est)
        W_est[np.abs(W_est) < w_threshold] = 0
        output = self.model.forward(self.X)
        return W_est, output