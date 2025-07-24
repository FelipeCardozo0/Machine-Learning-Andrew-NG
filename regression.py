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
