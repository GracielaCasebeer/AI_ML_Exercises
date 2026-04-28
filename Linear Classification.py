# Predicts whether a patient diagnosis of breast tissue is malignant or benign
# This is a classic application of machine learning.
# Deep learning has made great progress in this field.

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load the data
data = load_breast_cancer()

# Check the type of data
print(type(data))

# The bunch object acts like a dictionary where you can treat the keys
# like attributes
data.keys()
print(data.keys())

# The 'data' attribute means the input data
# We should have 569 samples and 30 input features
print(data.data.shape)

# The 'target' attribute contains just 0s and 1s
# This is a one-dimensional array, but when you have K targets, they are
# labeled 0..K-1
print(data.target)

# The target_names tells us the meaning of the 0s and 1s
print(data.target_names)

# Confirm that the length of Y = length of X, in this case: 569
print(data.target.shape)

# Determine the meaning of each of the 30 features
print(data.feature_names)

# Split the data into train and test sets. This lets us simulate how our
# model will perform in the future.
X_train, X_test, y_train, y_test = train_test_split(
    data.data, data.target, test_size = 0.33)
N, D = X_train.shape

# Scale the data
# Fit calculates the mean and variance
# Transform standardizes the data
# We only calculate th mean and variance (fit) in the train data, but not on
# the test data to prevent biasing the model. We want the test data to be a
# completely new and a "surprise" set for our model.
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Build the model
model = nn.Sequential(
    nn.Linear(D, 1),
    nn.Sigmoid()
)

# Loss and optimizer
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters())

# Convert data into torch tensors
X_train = torch.from_numpy(X_train.astype(np.float32))
X_test = torch.from_numpy(X_test.astype(np.float32))
y_train = torch.from_numpy(y_train.astype(np.float32).reshape(-1, 1))
y_test = torch.from_numpy(y_test.astype(np.float32).reshape(-1, 1))

# Train the model
n_epochs = 1000
train_losses = np.zeros(n_epochs)
test_losses = np.zeros(n_epochs)
train_accuracies = np.zeros(n_epochs)
test_accuracies = np.zeros(n_epochs)
for it in range(n_epochs):
    # Zero the parameter gradients
    optimizer.zero_grad()

    # Forward pass
    outputs = model(X_train)
    loss = criterion(outputs, y_train)

    # Backward and optimize
    loss.backward()
    optimizer.step()

    # Get test loss by doing the forward pass with X_test instead of X_train
    outputs_test = model(X_test)
    loss_test = criterion(outputs_test, y_test)

    # Save both losses, for train and test
    train_losses[it] = loss.item()
    test_losses[it] = loss_test.item()

    if (it + 1) % 50 == 0:
        print(f'Epoch {it + 1}/{n_epochs}, Train Loss: {loss.item():.4f}, '
              f'Test Loss: {loss_test.item():.4f}')

    # # Get accuracy for train and test data
    # with torch.no_grad():
    #     p_train = model(X_train)
    #     p_train = np.round(p_train.numpy())
    #     train_acc = np.mean(y_train.numpy() == p_train)
    #
    #     # Do the same procedure from above to the test data
    #     p_test = model(X_test)
    #     p_test = np.round(p_test.numpy())
    #     test_acc = np.mean(y_test.numpy() == p_test)
    #
    #     # Save both accuracies, for train and test
    #     train_accuracies[it] = train_acc
    #     test_accuracies[it] = test_acc
    #
    #     if (it + 1) % 50 == 0:
    #         print(f'Epoch {it + 1}/{n_epochs}, Train Accuracy: {train_acc:.4f}, '
    #               f'Test Accuracy: {test_acc:.4f}')

# Graphing the train and test losses per iteration
plt.plot(train_losses, label='Train Loss')
plt.plot(test_losses, label='Test Loss')
plt.legend()
plt.show()

# # Graphing the train and test accuracies per iteration
# plt.plot(train_accuracies, label='Train Accuracy')
# plt.plot(test_accuracies, label='Test Accuracy')
# plt.legend()
# plt.show()

# Get final accuracy
with torch.no_grad():
    # Get predictions from train data
    p_train = model(X_train)
    # Convert p_train from a torch tensor to a numpy array. Also convert
    # probabilities to 0s and 1s by rounding them.
    p_train = np.round(p_train.numpy())
    # Convert y_train from a torch tensor to a numpy array. Then, compare
    # element by element the y_train and p_train arrays. If they are equal
    # we get a 1; otherwise, we get a 0. Finally, we take the mean of the
    # comparisons. This is equivalent to calculating (# true)/(# total)
    train_acc = np.mean(y_train.numpy() == p_train)

    # Do the same procedure from above to the test data
    p_test = model(X_test)
    p_test = np.round(p_test.numpy())
    test_acc = np.mean(y_test.numpy() == p_test)
print(f"Train Accuracy: {train_acc}, Test Accuracy: {test_acc}")
