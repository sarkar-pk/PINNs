#################   ODE with hard enforced boundary conditions  #######
##  d2y/dx2 + pi^2 * sin(pi * x)                    ############
## with boundary conditions y(0) = y(1) = 0              ####### 
## having analytical solution: y = sin(pi * x)             #####
#######################################################################
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# Setting random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# 1. Neural Network Architecture for PINN

model_sequence = nn.Sequential(       # 1 input neuron, 4 hidden layers, 1 output 
    nn.Linear( 1, 20), nn.Tanh(),    # With Tanh activation 
    nn.Linear(20, 20), nn.Tanh(),
    nn.Linear(20, 20), nn.Tanh(),
    nn.Linear(20, 20), nn.Tanh(),
    nn.Linear(20,  1)
)
def model(x):
    return x*(1-x)*model_sequence(x)     # BC is enforced with the equation
    # Instantiate Optimizer
optimizer = optim.Adam(model_sequence.parameters(), lr=1e-3)
# 2. Physics-Informed Residual Computation
def compute_residual(model, x_collocation):
    x_collocation.requires_grad_(True)
    y = model(x_collocation)

    # First derivative dy/dx
    dy_dx = torch.autograd.grad(
        outputs=y,
        inputs=x_collocation,
        grad_outputs=torch.ones_like(y),
        create_graph=True,
        retain_graph=True
    )[0]

    # Second derivative d2y/dx2
    d2y_dx2 = torch.autograd.grad(
        outputs=dy_dx,
        inputs=x_collocation,
        grad_outputs=torch.ones_like(dy_dx),
        create_graph=True,
        retain_graph=True
    )[0]

    # ODE residual calculation: d2y/dx2 + pi^2 * sin(pi * x) = 0
    pi = torch.pi
    residual = d2y_dx2 + (pi**2) * torch.sin(pi * x_collocation)
    return residual

# 3. define loss function
def mse_loss(y_pred, y_exact):
    return torch.mean((y_pred - y_exact)**2)

# 4. Exact Analytical Solution
def exact_solution(x):
    return torch.sin(torch.pi * x)

# 5. Main Training 

# Problem domain: x in [0, 1]
x_min, x_max = 0.001, 0.999    # Since BC is already enforced, leaving the BC for training 

# Boundary Conditions (BC): y(0) = 0, y(1) = 0
x_bc = torch.tensor([[0.0], [1.0]], dtype=torch.float32)
y_bc = torch.tensor([[0.0], [0.0]], dtype=torch.float32)

# Collocation points inside the domain (0, 1)
N_collocation = 100
x_collocation = torch.linspace(x_min, x_max, N_collocation, dtype=torch.float32).unsqueeze(1)



epochs = 10000
#loss_history = []
#loss_bc_plot = []
loss = []

print("Training started")
for epoch in range(1, epochs + 1):
    optimizer.zero_grad()

    # Boundary loss: MSE between y_pred(x_bc) and y_bc
#    y_bc_pred = model(x_bc)
#    loss_bc = mse_loss(y_bc_pred, y_bc)

    # Physics (ODE) loss: MSE of residual at collocation points
    residual = compute_residual(model, x_collocation)
    loss_pde = mse_loss(residual, torch.zeros_like(residual))

#    total_loss = loss_bc + loss_pde  # useed 1:1 loss, no weightage

    # Backpropagation and Optimization
    loss_pde.backward()
    optimizer.step()
    loss.append(loss_pde.item())
#    loss_bc_plot.append(loss_bc.item())

    if epoch % 1000 == 0 or epoch == 1:
        print(f"Epoch {epoch:5d}/{epochs} | Total Loss: {loss_pde.item():.6e} ")

print("Training Completed")

# 5. Evaluation & Plotting
x_test = torch.linspace(0, 1, 200, dtype=torch.float32).unsqueeze(1)

with torch.no_grad():
    y_pred = model(x_test)
    y_true = exact_solution(x_test)

# Calculate L2 relative error
l2_error = torch.norm(y_pred - y_true) / torch.norm(y_true)
print(f"\nRelative L2 Error: {l2_error.item():.6e}")

# Plotting results
fig, axes = plt.subplots(3, 1, figsize=(6, 10))

# Plot 1: PINN Prediction vs Analytical Solution
axes[0].plot(x_test, y_true, 'r-', linewidth=2, label=r'Exact: $\sin(\pi x)$')
axes[0].plot(x_test, y_pred, 'b--', linewidth=2, label=r'PINN Prediction')
axes[0].scatter(x_bc, y_bc, color='black', zorder=5, label='Boundary Points')
axes[0].set_title(r'Solution to $\frac{d^2y}{dx^2} + \pi^2 \sin(\pi x) = 0$')
axes[0].set_xlabel('x')
axes[0].set_ylabel('y(x)')
axes[0].grid(True)
axes[0].legend()

# Plot 2: Loss parameters
axes[1].semilogy(range(1, epochs + 1), loss, label='Loss', color='green')
axes[1].set_title('Training Loss History')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Loss (log scale)')
axes[1].grid(True)
axes[1].legend()

# Plot 3: Absolute error
axes[2].plot(x_test, abs(y_true-y_pred), 'r-', linewidth=2, label=r'$\|y_{true} - y_{pred}\|$')
axes[2].set_xlabel('x')
axes[2].set_ylabel(r'$\|y_{true} - y_{pred}\|$')
axes[2].grid(True)
axes[2].legend()

plt.tight_layout()
# plt.savefig('pinn_solution.png', dpi=300)
plt.show()
