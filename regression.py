import numpy as np

# x_train is the input variable (size in 1000 square feet)
# y_train is the target (price in 1000s of dollars)
x_train = np.array([1.0, 2.0])
y_train = np.array([300.0, 500.0])
print(f"x_train = {x_train}")
print(f"y_train = {y_train}")

# x_train = [1.,2.]
# y_train = [300.,500]

# m is the number of training examples
print(f"x_train.shape: {x_train.shape}")  # This tells you x_train is a 1D array with 2 elements. # x_train.shape: (2,)
m = x_train.shape[0]
print(f"Number of training examples is: {m}")
m = len(x_train)
print(f"Number of training examples is: {m}")

# x_train.shape: (2,)
# Number of training examples is: 2'

x_train = np.array([1,2])
y_train = np.array([300,500])


i = 0 # Change this to 1 to see (x^1, y^1)

x_i = x_train[i]
y_i = y_train[i]
print(f"(x^({i}), y^({i})) = ({x_i}, {y_i})")
# (x^(0), y^(0)) = (1, 300)

import matplotlib.pyplot as plt  # imports the plotting library

x_train = [1.0, 1.5, 2.0, 2.5]   # example house sizes in 1000 sqft
y_train = [150, 200, 250, 300]   # corresponding house prices in 1000s of dollars

plt.scatter(x_train, y_train, marker='x', c='r')  # plot red 'x' markers for each data point
plt.title("Housing Prices")                       # set the title of the plot
plt.ylabel('Price (in 1000s of dollars)')         # label the y-axis
plt.xlabel('Size (1000 sqft)')                    # label the x-axis
plt.show()                                        # display the plot

x^0: f_wb = w*x[0] +b
# to create hundreds of function output, use a for loop
def compute_model_output(x, w, b):
    """
    Computes the prediction of a linear model y = wx + b
    Args:
      x (ndarray (m,)): input data, m examples
      w, b (scalars): model parameters (weight and bias)
    Returns:
      f_wb (ndarray (m,)): predicted values
    """
    m = x.shape[0]               # number of data points
    f_wb = np.zeros(m)           # initialize output array with zeros
    for i in range(m):           # loop over each data point
        f_wb[i] = w * x[i] + b   # apply linear model: y = wx + b

    return f_wb                  # return predictions


import numpy as np                        # imports NumPy for numerical operations
import matplotlib.pyplot as plt          # imports Matplotlib for plotting

w = 100
b = 50

x_train = np.array([1.0, 1.5, 2.0, 2.5])  # house sizes in 1000 sqft (input features)
y_train = np.array([150, 200, 250, 300])  # house prices in 1000s of dollars (target values)

tmp_f_wb = compute_model_output(x_train, w, b)  # predict prices using the linear model y = wx + b

plt.plot(x_train, tmp_f_wb, c='b', label='Our Prediction')  
# plots the model’s predicted values as a blue line

plt.scatter(x_train, y_train, marker='x', c='r', label='Actual Values')  
# plots the actual training data as red 'x' markers

plt.title("Housing Prices")                         # sets the plot title
plt.ylabel('Price (in 1000s of dollars)')           # labels the y-axis
plt.xlabel('Size (1000 sqft)')                      # labels the x-axis
plt.legend()                                        # shows legend to distinguish model vs. data
plt.show()                                          # renders the plot window

# example of a house with 1200sqft

w = 200                         
b = 100    
x_i = 1.2
cost_1200sqft = w * x_i + b    

print(f"${cost_1200sqft:.0f} thousand dollars") # $340 thousand dollars
