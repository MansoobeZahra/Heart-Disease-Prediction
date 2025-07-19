import numpy as np
from model_utils import compute_cost, compute_gradient, gradient_descent, predict
def compute_cost_reg(X, y, w, b, lambda_ = 1):
   
    m, n = X.shape
    
    # Calls the compute_cost function that you implemented above
    cost_without_reg = compute_cost(X, y, w, b) 
    
    # You need to calculate this value
    reg_cost = 0.
    
    for j in range(n):
              reg_cost_j = w[j] ** 2
              reg_cost = reg_cost + reg_cost_j
    reg_cost = (lambda_/(2 * m)) * reg_cost
    
    # Add the regularization cost to get the total cost
    total_cost = cost_without_reg + reg_cost
    return total_cost
def compute_gradient_reg(X, y, w, b, lambda_ = 1): 
    m, n = X.shape
    dj_db, dj_dw = compute_gradient(X, y, w, b)

    for j in range(n): 

          dj_dw_j_reg = (lambda_ / m ) *w[j]# Your code here to calculate the regularization term for dj_dw[j]

          dj_dw[j] = dj_dw[j] + dj_dw_j_reg
        
    return dj_db, dj_dw