####### 2D square plate of 2x2 units centered on (0,0). left side at 50K
####### right side at 100 K. top and bottom insulated.. 
####### map all with laplacian heat equation


import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# 1. PHYSICAL DOMAIN
x_min, x_max = -1.0, 1.0; y_min, y_max = -1.0, 1.0

T_left = 50.0; T_right = 100.0
T_ref = 75.0; T_scale = 25.0
# We train the network to predict normalized temperature:
# theta = (T - T_ref) / T_scale
#
# Actual temperature:
# T = T_ref + T_scale * theta

# 2. BOUNDARY CONDITION POINTS
N_bc = 200
# Left boundary: x = -1, T = 50 K
y_left = torch.linspace(y_min, y_max, N_bc).reshape(-1, 1)  # creates a column (200x1)
x_left = torch.ones(N_bc, 1) * x_min             # creates a column with all -1 values (200x1)
xy_left = torch.cat((x_left, y_left), dim=1)    # stacks x and y side by side (200X2)
theta_left = torch.ones(N_bc, 1) * ((T_left - T_ref) / T_scale)  # column of -1 (200x1) 50K=-1

# Right boundary: x = 1, T = 100 K
y_right = torch.linspace(y_min, y_max, N_bc).reshape(-1, 1) # column (200x1) y values on right
x_right = torch.ones(N_bc, 1) * x_max                       # column of all +1 (200x1)
xy_right = torch.cat((x_right, y_right), dim=1)             # stacks x,y side by side (200x2)
theta_right = torch.ones(N_bc, 1) * ((T_right - T_ref) / T_scale) # column of +1 (200x1) 100K=1

# Combine Dirichlet boundaries
xy_dirichlet = torch.cat((xy_left, xy_right), dim=0) # starck xy_l&r vertically (400x2) 
theta_dirichlet = torch.cat((theta_left, theta_right), dim=0) # stack vertically (400x1)

# 3. INSULATED TOP AND BOTTOM BOUNDARIES
# Insulation means dT/dy = 0
x_tb = torch.linspace(x_min, x_max, N_bc).reshape(-1, 1)
# Bottom boundary: y = -1
y_bottom = torch.ones(N_bc, 1) * y_min
xy_bottom = torch.cat((x_tb, y_bottom), dim=1)
# Top boundary: y = 1
y_top = torch.ones(N_bc, 1) * y_max
xy_top = torch.cat((x_tb, y_top), dim=1)

xy_neumann = torch.cat((xy_bottom, xy_top), dim=0)

# 4. COLLOCATION POINTS
N_collocation = 2000
def generate_collocation_points(N):
    x = x_min + (x_max - x_min) * torch.rand(N, 1)
    y = y_min + (y_max - y_min) * torch.rand(N, 1)
    xy = torch.cat((x, y), dim=1)        # create x,y collocation points of (N,2)
    return xy
xy_collocation = generate_collocation_points(N_collocation)

# 5. PINN NEURAL NETWORK
class PINN(nn.Module):
    def __init__(self,input_dim=2,hidden_dim=20,output_dim=1,num_layers=4):
        super(PINN, self).__init__()
        layers = []
        # Input layer
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.Tanh())
        # Hidden layers
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.Tanh())
        # Output layer
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.net = nn.Sequential(*layers)
        # Xavier initialization
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)
                nn.init.zeros_(layer.bias)
    def forward(self, xy):
        return self.net(xy)    
# Initialize PINN
model = PINN(input_dim=2,hidden_dim=20, output_dim=1,num_layers=4)

# 6. PDE RESIDUAL COMPUTATION
def compute_residual(model, xy):
    xy = xy.clone().detach().requires_grad_(True)  # copy of the xy tensor (N,2)
    # Predicted normalized temperature
    theta = model(xy)   # forward pass       (N,1)
    # First derivatives
    grad_theta = torch.autograd.grad(outputs=theta,inputs=xy,
                            grad_outputs=torch.ones_like(theta),create_graph=True)[0]  # (N, 2)
    theta_x = grad_theta[:, 0:1]   # takes column 0, i.e. dT/dx   (N,1)
    theta_y = grad_theta[:, 1:2]   # takes column 1, i.e. dT/dy   (N,1)
    # Second derivatives
    grad_theta_x = torch.autograd.grad(
        outputs=theta_x,
        inputs=xy,
        grad_outputs=torch.ones_like(theta_x),
        create_graph=True
    )[0]
    grad_theta_y = torch.autograd.grad(
        outputs=theta_y,
        inputs=xy,
        grad_outputs=torch.ones_like(theta_y),
        create_graph=True
    )[0]
    theta_xx = grad_theta_x[:, 0:1]
    theta_yy = grad_theta_y[:, 1:2]
    # Laplace equation
    residual = theta_xx + theta_yy
    return residual

# 7. BOUNDARY CONDITION LOSS
def compute_bc_loss(model):
    # Dirichlet boundary loss
    theta_pred = model(xy_dirichlet)
    loss_dirichlet = torch.mean((theta_pred - theta_dirichlet)**2)
    # Neumann boundary loss (insulated top and bottom)
    xy = xy_neumann.clone().detach().requires_grad_(True)
    theta = model(xy)
    grad_theta = torch.autograd.grad(
        outputs=theta,
        inputs=xy,
        grad_outputs=torch.ones_like(theta),
        create_graph=True
    )[0]
    theta_y = grad_theta[:, 1:2]
    loss_neumann = torch.mean(theta_y**2)
    # Total boundary loss
    loss_bc = loss_dirichlet + loss_neumann
    return loss_bc, loss_dirichlet, loss_neumann

# 8. TRAINING
learning_rate = 1e-3
epochs = 5000
optimizer = optim.Adam(model.parameters(),lr=learning_rate)
loss_history = []
loss_pde_history = []
loss_bc_history = []
for epoch in range(epochs):
    optimizer.zero_grad()
    # PDE loss
    residual = compute_residual(model, xy_collocation)
    loss_pde = torch.mean(residual**2)
    # Boundary loss
    loss_bc, loss_dir, loss_neu = compute_bc_loss(model)
    # Total loss
    loss = loss_pde + loss_bc
    # Backpropagation
    loss.backward()
    optimizer.step()
    # Store losses
    loss_history.append(loss.item())
    loss_pde_history.append(loss_pde.item())
    loss_bc_history.append(loss_bc.item())
    if epoch % 500 == 0:
        print(
            f"Epoch: {epoch:5d} | "
            f"Total: {loss.item():.6e} | "
            f"PDE: {loss_pde.item():.6e} | "
            f"BC: {loss_bc.item():.6e}"
        )
print("\nTraining completed!")

# 9. TEMPERATURE PREDICTION
N_plot = 150
x_plot = torch.linspace(x_min, x_max, N_plot)
y_plot = torch.linspace(y_min, y_max, N_plot)
X, Y = torch.meshgrid(x_plot, y_plot, indexing="ij")
xy_plot = torch.stack((X.reshape(-1), Y.reshape(-1)), dim=1)

model.eval()
with torch.no_grad():
    theta_pred = model(xy_plot)
    # Convert normalized temperature to Kelvin
    T_pred = T_ref + T_scale * theta_pred
T_pred = T_pred.cpu().numpy().reshape(N_plot, N_plot)

X = X.numpy()
Y = Y.numpy()

# 10. ANALYTICAL SOLUTION AND ERROR
T_exact = 75.0 + 25.0 * X
MSE_solution = np.mean((T_pred - T_exact)**2)

relative_L2_error = (np.linalg.norm(T_pred - T_exact) / np.linalg.norm(T_exact))
print("\nSolution MSE:", MSE_solution)
print("Relative L2 error:", relative_L2_error)

# 11. PLOT 2D TEMPERATURE DISTRIBUTION
fig, axes = plt.subplots(2, 2, figsize=(8, 8))
contour = axes[0,0].contourf(X, Y, T_pred, levels=50, cmap="inferno")
fig.colorbar(contour,ax=axes[0,0],label="Temperature (K)")
axes[0,0].set_xlabel("x (m)")
axes[0,0].set_ylabel("y (m)")
axes[0,0].set_title("PINN: 2D Temperature Distribution")
axes[0,0].axis("equal")
# 12. PLOT LOSS HISTORY
axes[1,0].semilogy(loss_pde_history,label="PDE Loss")
axes[1,0].semilogy(loss_bc_history,label="Boundary Loss")
axes[1,0].set_xlabel("Epoch")
axes[1,0].set_ylabel("MSE Loss")
axes[1,0].set_title("PINN Training Loss")
axes[1,0].legend()
axes[1,0].grid(True, which="both", alpha=0.3)
# 13. PLOT EXACT VS PINN SOLUTION
axes[0,1].contourf(X, Y, T_exact, levels=50,cmap="inferno")
fig.colorbar(contour,ax=axes[0,1], label="Exact Temperature (K)" )
axes[0,1].set_xlabel("x (m)")
axes[0,1].set_ylabel("y (m)")
axes[0,1].set_title("Analytical Temperature Distribution")
axes[0,1].axis("equal")
# 14. PLOT ABSOLUTE ERROR
error = np.abs(T_pred - T_exact)
contour = axes[1,1].contourf(X, Y, error, levels=50, cmap="viridis")
fig.colorbar(contour,ax=axes[1,1],label="Absolute Error (K)")
axes[1,1].set_xlabel("x (m)")
axes[1,1].set_ylabel("y (m)")
axes[1,1].set_title("Absolute Error: PINN vs Analytical Solution")
axes[1,1].axis("equal")

plt.show()
