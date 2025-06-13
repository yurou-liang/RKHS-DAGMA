# Kernel-Based Differentiable Learning of Non-Parametric Directed Acyclic Graphical Models

If you are interested in the theoretical background, please refer to our paper:  
[Kernel-Based Differentiable Learning of Non-Parametric Directed Acyclic Graphical Models](https://proceedings.mlr.press/v246/liang24a.html)

```bibtex
@InProceedings{pmlr-v246-liang24a,
  title = 	 {Kernel-Based Differentiable Learning of Non-Parametric Directed Acyclic Graphical Models},
  author =       {Liang, Yurou and Zadorozhnyi, Oleksandr and Drton, Mathias},
  booktitle = 	 {Proceedings of The 12th International Conference on Probabilistic Graphical Models},
  pages = 	 {253--272},
  year = 	 {2024},
  editor = 	 {Kwisthout, Johan and Renooij, Silja},
  volume = 	 {246},
  series = 	 {Proceedings of Machine Learning Research},
  month = 	 {11--13 Sep},
  publisher =    {PMLR},
  pdf = 	 {https://raw.githubusercontent.com/mlresearch/v246/main/assets/liang24a/liang24a.pdf},
  url = 	 {https://proceedings.mlr.press/v246/liang24a.html},
  abstract = 	 {Causal discovery amounts to learning a directed acyclic graph (DAG) that encodes a causal model. This model selection problem can be challenging due to its large combinatorial search space, particularly when dealing with non-parametric causal models. Recent research has sought to bypass the combinatorial search by reformulating causal discovery as a continuous optimization problem, employing constraints that ensure the acyclicity of the graph. In non-parametric settings, existing approaches typically rely on finite-dimensional approximations of the relationships between nodes, resulting in a score-based continuous optimization problem with a smooth acyclicity constraint. In this work, we develop an alternative approximation method by utilizing reproducing kernel Hilbert spaces (RKHS) and applying general sparsity-inducing regularization terms based on partial derivatives. Within this framework, we introduce an extended RKHS representer theorem. To enforce acyclicity, we advocate the log-determinant formulation of the acyclicity constraint and show its stability. Finally, we assess the performance of our proposed RKHS-DAGMA procedure through simulations and illustrative data analyses.}
}
</details>

## Contents
- `RKHS_DAGMA.py` - Supports continuous data for nonlinear models.
- `Toy_example.py` - Toy example with two nodes and common relationship from quadratic, qubic, inverse or sinus to illustrate RKHS-DAGMA.
- `Tuebingen` - Experiments of RKHS_DAGMA with real-world bivariate datasets.
- `experiments` - Experiments of RKHS_DAGMA with simulations compared with NOTEARS algorithms.

## Running a simple demo
The simplest way to illustrate RKHS-DAGMA is to run a toy example with two nodes where the functional relationship between two nodes is either quadratic, qubic, inverse or sinus:

```bash
cd RKHS/
python -m Toy_example -r quadratic
```

## Using RKHS-DAGMA
```bash
# X: Data matrix as a torch.tensor
device= torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_default_device(device)
torch.set_default_dtype(torch.double)
X = X.to(device)
eq_model = RKHS_DAGMA.RKHSDagma(X).to(device)
model = RKHS_DAGMA.RKHSDagma_nonlinear(eq_model)
```

## An Overview of RKHS-DAGMA
We propose a novel approximation method for the non-linear relationships by kernels: Let $k$ be a given kernel and $f_j$ denote the non-linear relationship between jth random variable $X_j$ with other random variables, to learn a sparse directed acyclic graph, $f_j$ can be represented by the following formula:

![Formula](Formula.png)

