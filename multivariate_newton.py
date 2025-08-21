import math
import autograd.numpy as np
from autograd import grad, hessian 

delta = 0.0001
epsilon = 0.001

# f must accept a multivariate x 

def optimize(x_0, f):
    """
    finds the local min of a multivariate function using Newton's Method

    Parameters
    --------------------------
    x_0: initial starting point, length n array
    f: function to optimize, must be a function of n vector  

    Returns
    --------------------------
    (x, f(x)), where x minimizes f

    """

    # initial vals
    x_vals = [x_0]
    x_t = x_0 

    dist = math.inf

    while dist > epsilon:
        hess_f = hessian(f) 
        grad_f = grad(f) 
        
        H = hess_f(x_t)
        H_inv = np.linalg.inv(H) 
        gradient = grad_f(x_t) 
        
        x_t = x_t - H_inv @ gradient
        dist = np.linalg.norm(x_t - x_vals[-1])
        x_vals.append(x_t)

    return f"(x, f(x)) = {(x_t, f(x_t))}"
