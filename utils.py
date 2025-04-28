import numpy as np
from scipy.optimize import minimize
from scipy.stats import chi2

# This file is a collection of the functions developed in various notebooks in this project

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
        Factorloadings

    specific_variance:  (p, p) matrix
        diagonal matrix of specific variances

    mu:                 (p, 1) vector 
        Means vector

    nsim:               float 
        How many observations should be simulated

    verbose:            boolean
        whether to print k and p

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


def calculate_objective(specific_variance, X_data, k, standardized=True):
    """
    Calculate the factor model maximum likelihood objective function, F.

    Parameters
    ---
    specific_variance : (p,) arraylike
        The specific variances for each variable
    
    X_data : (n, p) arraylike

    k: float
        Number of factors

    standardized :       boolean
        Whether to use correlation matrix (standardized variables) or the covariance matrix
        in calculations.

    Returns
    ---
    Objective function value: float
    """
    p = X_data.shape[1]

    # Step 1
    S = np.corrcoef(X_data.T) if standardized else np.cov(X_data.T)
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
    result = np.trace(internal) - np.log(np.linalg.det(internal)) - p

    return result

def factor_model_solution(X, k, x0_guess=None, standardized=True):
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

    standardized:       boolean
        Whether to use correlation matrix (standardized variables) or the covariance matrix
        in calculations.

    Returns
    ---
    tuple : (psi_hat, lambda_hat) 
        where
        psi_hat: (p,p) diagonal matrix. 
        lambda_hat: (p, k) factor loadings matrix
    
    """
    
    if x0_guess == None:
        x0_guess = np.ones(X.shape[1])

    # Optimize
    problem = minimize(fun=lambda x: calculate_objective(np.exp(x), X_data=X, k=k, standardized=standardized),
                       x0=x0_guess)
    
    psi_hat = np.diag(np.exp(problem.x))

    # Calculate lambda hat
    S = np.corrcoef(X.T) if standardized else np.cov(X.T)
    Psi_sq_inv = np.linalg.inv(psi_hat ** 0.5)
    S_star = Psi_sq_inv @ S @ Psi_sq_inv
    eigval, eigvec = np.linalg.eig(S_star)
    lambda_star = []
    for i in range(k):
        lambda_star.append(max(eigval[i] - 1, 0) ** 0.5 * eigvec[:,i])
    lambda_star = np.array(lambda_star).T
    lambda_hat = psi_hat ** 0.5 @ lambda_star

    return (psi_hat, lambda_hat)

def open_closed_data():
    """
    Returns
    ---
    Return the open/closed book data from Multivariate Statistics (p. 3-4), as an (n, p) matrix.
    """
    mechanics = [77, 63, 75, 55, 63, 53, 51, 59, 62, 64, 52, 55, 50 , 65, 31, 60, 44, 42, 62, 31, 44, 49, 12, 49, 54, 54, 44, 18, 46, 32, 30, 46, 40, 31, 36, 56, 46, 45, 42, 40, 23, 48, 41, 46, 46, 40, 49, 22, 35, 48, 31, 17, 49, 59, 37, 40, 35, 38, 43, 39, 62, 48, 34, 18, 35, 59, 41, 31, 17, 34, 46, 10, 46, 30, 13, 49, 18, 8, 23, 30, 3, 7, 15, 15, 5, 12, 5, 0]
    vectors = [82, 78, 73, 72, 63, 61, 67, 70, 60, 72, 64, 67, 50, 63, 55, 64, 69, 69, 46, 49, 61, 41, 58, 53, 49, 53, 56, 44, 52, 45, 69, 49, 27, 42, 59, 40, 56, 42, 60, 63, 55, 48, 63, 52, 61, 57, 49, 58, 60, 56, 57, 53, 57, 50, 56, 43, 35, 44, 43, 46, 44, 38,42, 51, 36, 53, 41, 52, 51, 30, 40, 46, 37, 34, 51, 50, 32, 42, 38, 24, 9, 51, 40, 38, 30, 30, 26, 40]
    algebra = [67, 80, 71, 63, 65, 72, 65, 68, 58, 60, 60, 59, 64, 58, 60, 56, 53, 61 ,61, 62, 52, 61, 61, 49, 56, 46, 55, 50, 65, 49, 50, 53, 54, 48, 51, 56, 57, 55, 54, 53, 59, 49, 49, 53, 46, 51, 45, 53, 47, 49, 50, 57, 47, 47, 49, 48, 41, 54, 38, 46, 36, 41, 50, 40, 46, 37, 43, 37, 52, 50, 47, 36, 45, 43, 50, 38, 31, 48, 36, 43, 51, 43, 43, 39, 44, 32, 15, 21]
    analysis = [67, 70, 66, 70, 70, 64, 65, 62, 62, 62, 63, 62, 55, 56, 57, 54, 53, 55, 57, 63, 62, 49, 63, 62, 47, 59, 61, 57, 50, 57, 52, 59, 61, 54, 45, 54, 49, 56, 49, 54, 53, 51, 46, 41, 38, 52, 48, 56, 54, 42, 54, 43, 39, 15, 28, 21, 51, 47, 34, 32, 22, 44, 47, 56, 48, 22, 30, 27, 35, 47, 29, 47, 15, 46, 25, 23, 45, 26, 48, 33, 47, 17, 23, 28, 36, 35, 20, 9]
    statistics = [81, 81, 81, 68, 63, 73, 68, 56, 70, 45, 54, 44, 63, 37, 73, 40, 53, 45, 45, 62, 46, 64, 67, 47, 53, 44, 36, 81, 35, 64, 45, 37, 61, 68, 51, 35, 32, 40, 33, 25, 44, 37, 34, 40, 41, 31, 39, 41, 33, 32, 34, 51, 26, 46, 45, 61, 50, 24, 49, 43, 42, 33, 29, 30, 29, 19, 33, 40, 31, 36, 17, 39, 30, 18, 31, 9, 40, 40, 15, 25, 40, 22, 18, 17, 18, 21, 20, 14]
    return np.array([mechanics, vectors, algebra, analysis, statistics]).T

def calculate_s(p, k):
    return 1/2 * (p - k) ** 2 - 1/2 * (p + k)

def factor_goodness_of_fit_test(X, k):
    """
    Calculate the p-value for the null hypothesis that k factors are sufficient to describe the data, 
    against the alternative that Sigma has no constraints.

    Parameters
    ---
    X :  (n, p) matrix
        Data matrix

    k :  Integer
        Number of factors to test for
    
    Returns
    ---
    p-value: float
        The p-value for the U statistic under the null hypothesis
    """
    
    psi_hat, _ = factor_model_solution(X, k)
    objective = calculate_objective(psi_hat[np.diag_indices_from(psi_hat)], X, k)
    n = X.shape[0]
    p = X.shape[1]

    n_mark = n - 1 - 1/6 * (2 * p + 5) - 2/3 * k

    U = objective * n_mark
    s = calculate_s(p, k)
    return chi2.sf(U, df=s)