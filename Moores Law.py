import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Fetch the data.
df = pd.read_csv("https://ourworldindata.org/grapher/transistors-"
                 "per-microprocessor.csv?v=1&csvType=filtered&useColumnShort"
                 "Names=true", storage_options = {'User-Agent':
                 'Our World In Data data fetch/1.0'})

# Drop unnecessary columns
data = df.drop(columns=['entity', 'code']).values

# X = only the values in the first data column [years]
# Y = only the values in the second data column [transistors per microprocessor]
# Convert X and Y into 2-D arrays of size N x D where D = 1
# Plot the data
X = data[:, 0].reshape(-1, 1)
Y = data[:, 1].reshape(-1, 1)
plt.scatter(X, Y)
plt.show()

# Data depicts an exponential function.
# Using logarithms to convert it into a linear function and plot the data
Y = np.log(Y)
plt.scatter(X, Y)
plt.show()

# Standardize/normalize X and Y to deal with a smaller range of numbers, then plot
mx = X.mean()
sx = X.std()
my = Y.mean()
sy = Y.std()
X = (X - mx) / sx
Y = (Y - my) / sy
plt.scatter(X, Y)
plt.show()

