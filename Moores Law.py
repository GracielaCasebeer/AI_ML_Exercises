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

# Create the linear regression model
model = nn.Linear(1, 1)

# Loss and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.7)

# PyTorch uses float32 by default
# Numpy creates float64 by default
inputs = torch.from_numpy(X.astype(np.float32))
targets = torch.from_numpy(Y.astype(np.float32))

print(type(inputs))

# Train the model
n_epochs = 100
losses = []
for it in range(n_epochs):
    # Zero the parameter gradients
    optimizer.zero_grad()

    # Forward pass
    outputs = model(inputs)
    loss = criterion(outputs, targets)

    # Keep the loss so we can plot it later
    losses.append(loss.item())

    # Backward and optimize
    loss.backward()
    optimizer.step()

    print(f'Epoch {it+1}/{n_epochs}, Loss: {loss.item():.4f}')

# Graphing loss per iteration
plt.plot(losses)
plt.show()

# Plot the graph
predicted = model(inputs).detach().numpy()
plt.plot(X, Y, 'ro', label='Original data')
plt.plot(X, predicted, label='Fitted line')
plt.legend()
plt.show()

# Test the efficacy of the model
# True values of (w, b) are (0.5, -1)
w = model.weight.data.numpy()
b = model.bias.data.numpy()
print('Slope and intercept of the linear regression:')
print(w, b)

# The logarithmic model is y' = wx' + b, where y' and x' are normalized.
# y' = (y - m_sub_y)/s_sub_y and x' = (x - m_sub_x)/s_sub_x.
# We recover the original model by substituting the values of y' and x' in
# the linear logarithmic model. After some algebraic manipulation (not shown
# here), we find that the slope of the line, which determines the rate of
# growth is given by a = w * (s_sub_y/s_sub_x)
a = w[0,0] * (sy/sx)
print(a)

# Now we need to go back to the original exponential equation C = C_o*r^t
# that we converted into linear by applying logarithms.
# log C = log r * t + log C_o is equivalent to y = ax + log C_o
# Therefore:
# y = log C, a = log r, and x = t
# We know the value of a (from the calculation above), so we can find the
# value of r by applying the exponential: r = e ^ a
# If t' is the time to double the transistor count: 2C = C_o * r ^ t'
# Dividing the two equations: 2C/C = 2 = r ^ (t' - t)
# Solving for (t' - t) = Delta(t) = (log 2)/(log r) = (log 2)/a, the time to
# double the transistor count.
print("Time to double: ", (np.log(2)/a))

# Therefore, Moore's law is true. Transistor count doubles every 2 years.

