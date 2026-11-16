#################   ODE with class function, can be called  ####################
## Hard enforced boundary conditions                ###############
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
# Can be called for other code
class PINN(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=20, output_dim=1, num_layers=4):
        super(PINN, self).__init__()

        layers = []
        # Input layer
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.Tanh())  # activation function Tanh

        # Hidden layers
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.Tanh())

        # Output layer
        layers.append(nn.Linear(hidden_dim, output_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return x*(1-x)*self.net(x)


# 2. Physics-Informed Residual Computation
def compute_residual(model, x_collocation):
    """
    Computes the ODE residual:   f = d2y/dx2 + pi^2 * sin(pi * x)
    """
    x_collocation.requires_grad_(True)

    # Forward pass: model output y(x)
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


# 3. Exact Analytical Solution
def exact_solution(x):
    return torch.sin(torch.pi * x)


# 4. Main Training 
def train_pinn():
    # Problem domain: x in [0, 1]
    x_min, x_max = 0.0, 1.0

    # Boundary Conditions (BC): y(0) = 0, y(1) = 0
    x_bc = torch.tensor([[x_min], [x_max]], dtype=torch.float32)
    y_bc = torch.tensor([[0.0], [0.0]], dtype=torch.float32)

    # Collocation points inside the domain (0, 1)
    N_collocation = 500
    x_collocation = torch.linspace(x_min+0.001, x_max-0.001, N_collocation, dtype=torch.float32).unsqueeze(1)

    # Instantiate Model, Loss Function, and Optimizer
    model = PINN(input_dim=1, hidden_dim=20, output_dim=1, num_layers=4)
    mse_loss = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    epochs = 5000
    loss_history = []

    print("Training started")
    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()

        # Physics (ODE) loss: MSE of residual at collocation points
        residual = compute_residual(model, x_collocation)
        loss_pde = mse_loss(residual, torch.zeros_like(residual))

        # Total loss (weighted sum if necessary; here 1:1 ratio works well)
        total_loss =  loss_pde

        # Backpropagation and Optimization
        total_loss.backward()
        optimizer.step()
        loss_history.append(total_loss.item())

        if epoch % 500 == 0 or epoch == 1:
            print(f"Epoch {epoch:5d}/{epochs} | Total Loss: {total_loss.item():.6e} ")

    print("Training Completed")

    # 5. Evaluation & Plotting
    model.eval()
    x_test = torch.linspace(x_min, x_max, 200, dtype=torch.float32).unsqueeze(1)
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
    axes[1].semilogy(range(1, epochs + 1), loss_history, label='Loss', color='purple')
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

if __name__ == "__main__":
    train_pinn()

