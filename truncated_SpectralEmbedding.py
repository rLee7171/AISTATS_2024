import numpy as np
import time
import torch
from scipy.sparse import diags, spdiags, eye
from scipy.linalg import svd
from torch_geometric.utils import to_scipy_sparse_matrix
from scipy.sparse.linalg import eigs
from scipy.linalg import eig

def truncated_spectral_embedding(G, dim, root_dir, dataset_name, iter=1, amb_dim=None,use_cache=False):
    # Set defaults
    if amb_dim is None:
        amb_dim = 2 * dim
    t1 = time.time()
    # Build Laplacians
    edge_index = G.edge_index
    G = to_scipy_sparse_matrix(edge_index=edge_index).tocsr()
    n = G.shape[0]
    d = G.sum(axis=0).A.flatten()
    sd = d.sum()

    # Symmetric normalized adjacency and Laplacian
    diagonal_matrix = diags(1/np.sqrt(d),0)
    snG = diagonal_matrix @ G @ diagonal_matrix
    snL = eye(n) - snG
    # Draw random basis in amb_dim
    B = np.random.randn(n, amb_dim)

    # Iterative improvement of B
    for j in range(iter):
        B = (eye(n) - snL) @ (B)

    B = np.hstack((B, np.sqrt(d).reshape(-1, 1)))
    # Form small eigenvalue problem for snL
    M = np.linalg.pinv(B) @ snL @ B
    
    D, V = eigs(M,amb_dim-1)
    idx = D.argsort()[::-1]
    D = D[idx]
    V = V[:,idx]
    V = B.dot(V)  # All eigenvectors
    lambda_vals = np.diag(D)  # All eigenvalues
    eigVects = diags(np.sqrt(1/d),0) @ eye(n) @ V[:,:dim] 
    lambda_vals = lambda_vals[0:dim]
    
    # Apply projection step
    eigVects = projection_step(eigVects, d, sd)

    t2 = time.time()
    runtimeInfo = {'total': t2 - t1}

    cache_dir_egval = root_dir + "/Cache/eigval_trun_embedding_" + str(dataset_name) + "_iter_"+ str(iter) + ".pt"
    cache_dir_egvec = root_dir + "/Cache/eigvec_trun_embedding_" + str(dataset_name) + "_iter_"+ str(iter) + ".pt"

    if use_cache:
        vectors = torch.load(cache_dir_egvec, map_location=torch.device('cpu'))
        values = torch.load(cache_dir_egval, map_location=torch.device('cpu'))
    else:
        vectors = torch.tensor(eigVects)
        values = torch.tensor(lambda_vals[:dim])
        vectors = vectors.to(torch.float32)
        values = values.to(torch.float32)

    return vectors, values, runtimeInfo

def Bfun(x, d, sd):
    return d * x - d * ((d.transpose() * x)/sd)

def projection_step(eigVects, d, sd):
    n = eigVects.shape[0]
    for i in range(eigVects.shape[1]):
        c = -np.dot(d, eigVects[:, i]) / np.dot(d, np.ones(n))
        eigVects[:, i] = eigVects[:, i] + c * np.ones(n)
        eigVects[:, i] = eigVects[:, i] / np.sqrt(eigVects[:, i].transpose() @ Bfun(eigVects[:, i], d, sd) )

    return eigVects
