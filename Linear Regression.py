import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# Set number of data points desired
N = 20

# Generate random data on the x-axis in (-5, 5)
X = (np.random.random(N)*10) - 5

# Generate a line plus some noise
Y = (0.5 * X) - 1 + np.random.randn(N)

# Plot the data
plt.scatter(X, Y)
plt.show()

# Create the linear regression model
model = nn.Linear(1, 1)

# Loss and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

# Set ML data shape (num_samples x num_dimensions)
# Both X and Y become N x 1 matrices
X = X.reshape(N, 1)
Y = Y.reshape(N, 1)

# PyTorch uses float32 by default
# Numpy creates float64 by default
inputs = torch.from_numpy(X.astype(np.float32))
targets = torch.from_numpy(Y.astype(np.float32))

print(type(inputs))

# Train the model
n_epochs = 30
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
# plt.scatter(X, Y, label='Original data')
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