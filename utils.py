import numpy as np
from scipy.optimize import minimize

def sim_factor_model_slow(loadings, specific_variance, mu, nsim=1, verbose=True):
    """
    Equal to sim_factor_model. This is kept here as this method doesn't use vectorisation,
    and is thus easier to understand.
    The other version uses vectorisation to significantly increase the speed at which it can simulate.
    """
    k = loadings.shape[1]
    p = len(specific_variance)
    if verbose:
        print(f"{k=} {p=}")
    X = []
    for _ in range(nsim):
        factor_vector = np.random.multivariate_normal(np.zeros(k), np.eye(k))
        u = np.random.multivariate_normal(np.zeros(p), np.diag(specific_variance))

        X.append(loadings @ factor_vector + u + mu)

    return np.array(X)

def sim_factor_model(loadings, specific_variance_vec, mu, nsim=1, verbose=True):
    """
    Parameters
    ---
    loadings:           (p, k) matrix

    specific_variance:  (p, p) diagonal matrix of specific variances

    mu:                 (p, 1) vector of means

    nsim:               flaot How many observations should be simulated

    verbose:            Boolean, whether to print k and p

    Returns
    ---
        (n, p) matrix of observations from the specified factor model

    """
    k = loadings.shape[1]
    p = len(specific_variance_vec)

    if verbose:
        print(f"{k=} {p=}")

    # Generate nsim factor vectors from a standard normal distribution
    factor_vectors = np.random.normal(loc=0, scale=1, size=(nsim, k))

    # Generate nsim specific errors using element-wise normal sampling
    specific_errors = np.random.normal(loc=0, scale=np.sqrt(specific_variance_vec), size=(nsim, p))

    # Compute observations
    X = factor_vectors @ loadings.T + specific_errors + mu

    return X


def calculate_objective(specific_variance, X_data, k):
    """
    Calculate the factor model maximum likelihood objective function.

    Parameters
    ---
    specific_variance:  (p,) arraylike
        The specific variances for each variable
    
    X_data: (n, p) arraylike

    k: float
        Number of factors

    Returns
    ---
    Objective function value: float
    """

    # Step 1
    S = np.cov(X_data.T)
    Psi = np.diag(specific_variance)
    Psi_sq_inv = np.linalg.inv(Psi ** 0.5)
    S_star = Psi_sq_inv @ S @ Psi_sq_inv

    # Step 2
    eigval, eigvec = np.linalg.eig(S_star)

    # Step 3
    lambda_star = []
    for i in range(k):
        lambda_star.append(max(eigval[i] - 1, 0) ** 0.5 * eigvec[:,i])
    lambda_star = np.array(lambda_star).T

    # Step 4
    lambda_hat = Psi ** 0.5 @ lambda_star

    # Step 5
    internal = np.linalg.inv(lambda_hat @ lambda_hat.T + Psi) @ S
    result = np.trace(internal) - np.log(np.linalg.det(internal))

    return result

def factor_model_solution(X, k, x0_guess=None):
    """
    Optimize the factor model w.r.t. psi, and calculate psi hat and lambda hat.

    Parameters
    ---
    X: (n, p) arraylike
        data matrix

    k: integer
        Number of factors

    x0_guess: (p,) arraylike
        An inital guess for the minimization algorithm
        If x0_guess is None,
        then default guess is specific variance 1 for all variables.

    Returns
    ---
    Tuple (psi_hat, lambda_hat) where:
    
    psi_hat: (p,p) diagonal matrix

    lambda_hat: (p, k) factor loadings matrix
    """
    
    if x0_guess == None:
        x0_guess = np.ones(X.shape[1])

    # Optimize
    problem = minimize(fun=lambda x: calculate_objective(np.exp(x), X_data=X, k=k),
                       x0=x0_guess)
    
    psi_hat = np.diag(np.exp(problem.x))

    # Calculate lambda hat
    S = np.cov(X.T)
    Psi_sq_inv = np.linalg.inv(psi_hat ** 0.5)
    S_star = Psi_sq_inv @ S @ Psi_sq_inv
    eigval, eigvec = np.linalg.eig(S_star)
    lambda_star = []
    for i in range(k):
        lambda_star.append(max(eigval[i] - 1, 0) ** 0.5 * eigvec[:,i])
    lambda_star = np.array(lambda_star).T
    lambda_hat = psi_hat ** 0.5 @ lambda_star

    return (psi_hat, lambda_hat)