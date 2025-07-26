# Mean Square Error 

def compute_cost(x, y, w, b): 
    """
    Computes the cost function for linear regression.
    
    Args:
      x (ndarray (m,)): Input features (m examples)
      y (ndarray (m,)): Target values
      w, b (scalars): Model parameters
    
    Returns:
      total_cost (float): Mean squared error cost
    """
    m = x.shape[0]             # Number of training examples
    cost_sum = 0               # Accumulator for total squared error
    
    for i in range(m): 
        f_wb = w * x[i] + b    # Linear prediction for example i
        cost = (f_wb - y[i])**2  # Squared error for example i
        cost_sum += cost       # Accumulate squared error
    
    total_cost = (1 / (2 * m)) * cost_sum  # Final averaged cost with 1/2 factor
    
    return total_cost
  
plt.close('all') 
fig, ax, dyn_items = plt_stationary(x_train, y_train)
updater = plt_update_onclick(fig, ax, x_train, y_train, dyn_items)
