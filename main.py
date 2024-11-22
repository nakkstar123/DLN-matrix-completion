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
learning_rate = 0.009
tolerance = 1e-15
max_iterations = 1000

# Initialize d and N
d = 10  # Dimension of the matrices
N = 20 # Number of random matrices
initial_scale = 1/np.sqrt(d)

# Initialize lists to store results
x_12_values = []
x_21_values = []

# Initialize lists to store eigenvalues for convergent cases
eigenvalues_list = []

# Run the optimization 10 times
for _ in trange(50):
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
            # Only store the off-diagonal values and eigenvalues if converged
            if loss < tolerance:
                final_X = W_matrices[0]
                for i in range(1, N):
                    final_X = W_matrices[i] @ final_X 
                x_12_values.append(final_X[0, 1])
                x_21_values.append(final_X[1, 0])
                
                # Store the eigenvalues of the converged matrix
                eigenvalues = np.linalg.eigvals(final_X)
                eigenvalues_list.append(eigenvalues)
            break

        # Compute gradients
        grads = compute_gradient(W_matrices)
        
        # Update W_i matrices
        for i in range(N):
            W_matrices[i] -= learning_rate * grads[i]

        iteration += 1

# Plot x_12 vs x_21
plt.scatter(x_12_values, x_21_values, label='Data Points', s=10)
plt.xlabel(r'$x_{12}$')
plt.ylabel(r'$x_{21}$')
plt.title('Off-diagonal entries after convergence')
plt.grid(True)
plt.xlim(-4, 4)
plt.ylim(-4, 4)

# Add the graph of y = 1/x with better spacing
x_values = np.linspace(-4, 0, 100)  # Create a single range from -4 to 4
y_values = 1 / np.array(x_values)
plt.plot(x_values, y_values, color='red', label=r'$y = \frac{1}{x}$', linewidth=0.5)

x_values = np.linspace(0, 4, 100)  # Create a single range from -4 to 4
y_values = 1 / np.array(x_values)
plt.plot(x_values, y_values, color='red', linewidth=0.5)

plt.legend()
plt.show()

# Plot the spectrum of all convergent product matrices X in one plot
plt.figure()
for eigenvalues in eigenvalues_list:
    plt.scatter(np.real(eigenvalues), np.imag(eigenvalues), s=10)

plt.axhline(0, color='grey', lw=0.5, ls='--')
plt.axvline(0, color='grey', lw=0.5, ls='--')
plt.xlabel('Real Part')
plt.ylabel('Imaginary Part')
plt.title('Spectrum of Converged Product Matrices X')
plt.grid(True)
plt.legend()
plt.show()

real_parts = np.concatenate([np.real(eigenvalues) for eigenvalues in eigenvalues_list])  # Collect all real parts
plt.figure()
plt.hist(real_parts, bins=30, color='blue', alpha=0.7)  # Create histogram
plt.xlabel('Real Part of Eigenvalues')
plt.ylabel('Frequency')
plt.title('Histogram of Real Parts of Eigenvalues')
plt.grid(True)
plt.show()
