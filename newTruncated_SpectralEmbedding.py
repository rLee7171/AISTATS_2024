def compute_projected_eigenvectors(G, X):
    """
    This function projects the normalized Laplacian of a graph onto a subspace spanned by 
    the columns of matrix X, then computes the eigenvectors of the projected matrix.
    
    Parameters:
    -----------
    G : torch_geometric.data.Data or similar
        Graph object with edge_index attribute containing the edge connectivity
    X : numpy.ndarray
        Matrix whose columns span the subspace for projection
        
    Returns:
    --------
    x : numpy.ndarray
        Eigenvectors of the projected normalized Laplacian, with each column normalized to unit norm
    lambda_v : numpy.ndarray
        Corresponding eigenvalues of the normalized Laplacian restricted to the subspace,
        sorted in ascending order
    runtimeInfo : dict
        Dictionary containing timing information
    """
    t1 = time.time()
    
    # Ensure X is orthonormal using QR decomposition
    # Q contains orthonormal columns that span the same subspace as X
    Q, R = qr(X, mode='economic')
    
    # Convert graph to scipy sparse matrix format
    G = to_scipy_sparse_matrix(edge_index=G.edge_index).tocsr()
    n = G.shape[0]
    
    # Compute the standard Laplacian matrix L = I - A
    # where A is the adjacency matrix and I is the identity matrix
    L = eye(n) - G
    
    # Compute degree vector (sum of each row in adjacency matrix)
    d = G.sum(axis=1).A.flatten()
    
    # Compute D^(-1/2) for normalized Laplacian
    # D is the diagonal degree matrix, D^(-1/2) has 1/sqrt(d_i) on diagonal
    d_inv_sqrt = diags(1 / np.sqrt(d))
    
    # Compute the normalized Laplacian: L_norm = D^(-1/2) * L * D^(-1/2)
    # The normalized Laplacian has eigenvalues in [0, 2] and is symmetric
    L_norm = d_inv_sqrt @ L @ d_inv_sqrt
    
    # Project the normalized Laplacian onto the subspace spanned by X
    # This reduces the problem size from n x n to k x k where k is the number of columns in X
    L_proj = Q.T @ L_norm @ Q
    
    # Compute eigenvalues and eigenvectors of the projected matrix
    # These are the eigenvalues of the normalized Laplacian restricted to the subspace
    lambda_v, x = eigh(L_proj)
    
    # Transform eigenvectors back to the original space
    # The eigenvectors are currently in the projected space, we need to map them back
    x = Q @ x
    
    # Normalize each column of x to have unit norm
    # This ensures all eigenvectors have the same magnitude
    x_norms = np.linalg.norm(x, axis=0)
    x = x / x_norms[np.newaxis, :]
    
    # Sort eigenvalues and eigenvectors in ascending order
    # For Laplacian matrices, the smallest eigenvalue is always 0
    # The second smallest eigenvalue (algebraic connectivity) is important for graph connectivity
    idx = lambda_v.argsort()
    lambda_v = lambda_v[idx]
    x = x[:, idx]
    
    t2 = time.time()
    runtimeInfo = {'total': t2 - t1}
    
    return x, lambda_v, runtimeInfo

def create_power_iteration_subspace(G, k, num_iterations):
    """
    Create subspace using power iteration with normalized adjacency matrix.
    
    Parameters:
    -----------
    G : graph object
        Graph with edge_index
    k : int
        Dimension of initial random matrix (2k for our case)
    num_iterations : int
        Number of power iterations (j = 2, 4, 8, 16, 32)
    
    Returns:
    --------
    X : numpy.ndarray
        Subspace matrix after power iterations
    """
    # Convert to adjacency matrix
    A = to_scipy_sparse_matrix(edge_index=G.edge_index).tocsr()
    n = A.shape[0]
    
    # Compute degree matrix D^(-1/2)
    d = A.sum(axis=1).A.flatten()
    d_inv_sqrt = diags(1 / np.sqrt(d))
    
    # Normalized adjacency: D^(-1/2) A D^(-1/2)
    A_norm = d_inv_sqrt @ A @ d_inv_sqrt
    
    # Initialize with random matrix
    X = np.random.randn(n, k)
    
    # Power iteration: X = (D^(-1/2) A D^(-1/2))^j X
    # Use sparse matrix-vector multiplications, NOT matrix powering
    for _ in range(num_iterations):
        X = A_norm @ X
        
    return X

def preprocess_feature_matrix(X, target_dim=None):
    """
    Preprocess feature matrix to handle rank deficiency and dimension reduction.
    
    Parameters:
    -----------
    X : numpy.ndarray
        Input feature matrix (may not be full rank)
    target_dim : int, optional
        Target dimension (2k in our case). If None, keep all dimensions.
    
    Returns:
    --------
    X_processed : numpy.ndarray
        Orthonormal basis for the column space of X
    rank_info : dict
        Information about the rank and dimension reduction
    """
    # Compute SVD to handle rank deficiency
    U, s, Vt = np.linalg.svd(X, full_matrices=False)
    
    # Determine effective rank (remove near-zero singular values)
    tol = 1e-12
    effective_rank = np.sum(s > tol)
    
    # Take orthonormal columns corresponding to non-zero singular values
    X_processed = U[:, :effective_rank]
    
    # If dimension is larger than target, reduce to target_dim
    if target_dim is not None and effective_rank > target_dim:
        X_processed = X_processed[:, :target_dim]
        final_dim = target_dim
    else:
        final_dim = effective_rank
    
    rank_info = {
        'original_shape': X.shape,
        'original_rank': effective_rank,
        'final_dim': final_dim,
        'dimension_reduced': effective_rank > final_dim if target_dim else False
    }
    
    return X_processed, rank_info
