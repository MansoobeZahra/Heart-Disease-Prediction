# model_utils.py

import numpy as np
import math
# Sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Cost function
def compute_cost(X, y, w, b, *argv):
    """
    Computes the cost over all examples
    Args:
      X : (ndarray Shape (m,n)) data, m examples by n features
      y : (ndarray Shape (m,))  target value 
      w : (ndarray Shape (n,))  values of parameters of the model      
      b : (scalar)              value of bias parameter of the model
      *argv : unused, for compatibility with regularized version below
    Returns:
      total_cost : (scalar) cost     """

    m, n = X.shape
    
    total_cost = 0
    for i in range(m):
        f_wb_i = np.dot(X[i], w) + b
        z= 1 / (1 + np.exp(-f_wb_i))
        loss = ( y[i] * np.log(z)) +( (1-y[i]) * np.log(1-z))
        loss=-loss
        total_cost=total_cost+loss
    total_cost = total_cost / ( m)
    return total_cost
# Gradient descent
def compute_gradient(X, y, w, b, *argv): 
    m, n = X.shape
    dj_dw = np.zeros(w.shape)
    dj_db = 0.
    for i in range(m):
        f_wb_i = np.dot(X[i], w) + b
        f_wb= 1 / (1 + np.exp(-f_wb_i))
        dj_db_i = f_wb - y[i]
        dj_db += dj_db_i    
        for j in range(n):
            dj_dw_ij = (f_wb - y[i])* X[i][j]
            dj_dw[j] += dj_dw_ij
    dj_dw = dj_dw / m
    dj_db = dj_db/m
    return dj_db, dj_dw
# Gradient Descent loop
def gradient_descent(X, y, w_in, b_in, cost_function, gradient_function, alpha, num_iters, lambda_): 
    
    # number of training examples
    m = len(X)
    
    # An array to store cost J and w's at each iteration primarily for graphing later
    J_history = []
    w_history = []
    
    for i in range(num_iters):

        # Calculate the gradient and update the parameters
        dj_db, dj_dw = gradient_function(X, y, w_in, b_in, lambda_)   

        # Update Parameters using w, b, alpha and gradient
        w_in = w_in - alpha * dj_dw               
        b_in = b_in - alpha * dj_db              
       
        # Save cost J at each iteration
        if i<100000:      # prevent resource exhaustion 
            cost =  cost_function(X, y, w_in, b_in, lambda_)
            J_history.append(cost)

        # Print cost every at intervals 10 times or as many iterations if < 10
        if i% math.ceil(num_iters/10) == 0 or i == (num_iters-1):
            w_history.append(w_in)
            print(f"Iteration {i:4}: Cost {float(J_history[-1]):8.2f}   ")
        
    return w_in, b_in, J_history, w_history 

def predict(X, w, b): 
    m, n = X.shape   
    p = np.zeros(m)
   
    for i in range(m):   
        z_wb = None
        f_wb_i = np.dot(X[i], w) + b
        f_wb= 1 / (1 + np.exp(-f_wb_i))
       
        p[i] = f_wb >= 0.5
        
    return p
