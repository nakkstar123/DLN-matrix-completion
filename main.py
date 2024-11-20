import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange

# Function to compute the loss
def compute_loss(X):
    return 0.5 * ((X[0, 0] - 1)**2 + (X[1, 1] - 1)**2)

# Function to compute the gradient of the loss w.r.t. matrices
def compute_gradient(W_matrices):
    N = len(W_matrices)
    d = W_matrices[0].shape[0]  # Assuming all matrices are d x d

    X = W_matrices[0]
    for i in range(1, N):
        X = W_matrices[i] @ X
    
    grad_X = np.zeros((2, 2))  # Initialize grad_X as a 2 x 2 matrix
    for i in range(2):
        grad_X[i, i] = X[i, i] - 1  # Set diagonal entries

    grads = []
    for j in range(N):
        grad_left = np.eye(d)  
        grad_right = np.eye(2) 
        for k in range(j):
            grad_right = grad_right @ W_matrices[k].T
        for k in range(j+1, N):
            grad_left = grad_left @ W_matrices[k].T 
        
        if j == 0:
            grad = grad_left @ grad_X
        elif j == N-1:
            grad = grad_X @ grad_right
        else:
            grad = grad_left @ grad_X @ grad_right
        grads.append(grad)
    
    return grads

# Gradient descent parameters
learning_rate = 0.05
tolerance = 1e-15
max_iterations = 1000

# Initialize d and N
d = 2  # Dimension of the matrices
N = 10 # Number of random matrices
initial_scale = 1/np.sqrt(d)

# Initialize lists to store results
x_12_values = []
x_21_values = []

# Run the optimization 10 times
for _ in trange(1000):
    # Initialize a list of N random matrices W_i
    W_matrices = [np.random.randn(d, 2) * initial_scale] + [np.random.randn(d, d) * initial_scale for _ in range(N-2)] + [np.random.randn(2, d) * initial_scale]

    loss_history = []
    iteration = 0

    while True:
        # Compute current X and loss
        X = W_matrices[0]
        for i in range(1, N):
            X = W_matrices[i] @ X
        loss = compute_loss(X)
        loss_history.append(loss)
        
        # Check stopping condition
        if loss < tolerance or iteration >= max_iterations:
            print(f"Converged in {iteration} iterations")
            break

        # Compute gradients
        grads = compute_gradient(W_matrices)
        
        # Update W_i matrices
        for i in range(N):
            W_matrices[i] -= learning_rate * grads[i]

        iteration += 1

    # Store the off-diagonal values after convergence
    final_X = W_matrices[0]
    for i in range(1, N):
        final_X = W_matrices[i] @ final_X 
    x_12_values.append(final_X[0, 1])
    x_21_values.append(final_X[1, 0])

# Plot x_12 vs x_21
plt.scatter(x_12_values, x_21_values)
plt.xlabel('X_12')
plt.ylabel('X_21')
plt.title('Off-diagonal entries after convergence')
plt.grid(True)
plt.show()
