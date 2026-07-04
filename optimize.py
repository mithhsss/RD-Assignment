# Here we are doing curve optimization 
from torch.optim import optimizer
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np 
import os

# In the first step we are loading the data points of x and y
print("Loading data from csv..")
if not os.path.exists("xy_data.csv"):
    print("Error:Data not found")
    exit()
data = pd.read_csv("xy_data.csv")
x_expected = torch.tensor(data['x'].values , dtype=torch.float32)
y_expected = torch.tensor(data['y'].values , dtype=torch.float32)

t_data = torch.tensor(np.linspace(6,60,len(data)),dtype = torch.float32)


# Step 2 here we design the model architecture (The weights)

theta_deg = torch.tensor([25.0],requires_grad=True)
M = torch.tensor([0.0],requires_grad=True)
X = torch.tensor([50.0],requires_grad=True)

# We are using L-BFGS optimizer to find the best values for our parameters
optimizer = optim.LBFGS([theta_deg, M, X], lr=0.1, max_iter=20)
criterion = nn.L1Loss()

epochs = 1000
print(f"Started optimization for {epochs} epochs using LBFGS")

# here we have the training loop

for epoch in range(epochs):
    def closure():
        optimizer.zero_grad()
        theta_rad = theta_deg * (np.pi / 180.0)

        # rotate back to get curve coordinates t and z_actual
        dx = x_expected - X
        dy = y_expected - 42.0
        t = dx * torch.cos(theta_rad) + dy * torch.sin(theta_rad)
        z_actual = -dx * torch.sin(theta_rad) + dy * torch.cos(theta_rad)

        z_model = torch.exp(M * torch.abs(t)) * torch.sin(0.3 * t)
        
        total_loss = criterion(z_model, z_actual)

        total_loss.backward()
        return total_loss

    optimizer.step(closure)

    # Re-evaluate loss for printing and saving
    with torch.no_grad():
        theta_deg.clamp_(0.001, 49.999)
        M.clamp_(-0.049, 0.040)
        X.clamp_(0.001, 99.999)
        
        theta_rad = theta_deg * (np.pi / 180.0)
        dx = x_expected - X
        dy = y_expected - 42.0
        t = dx * torch.cos(theta_rad) + dy * torch.sin(theta_rad)
        z_actual = -dx * torch.sin(theta_rad) + dy * torch.cos(theta_rad)
        z_model = torch.exp(M * torch.abs(t)) * torch.sin(0.3 * t)
        total_loss = criterion(z_model, z_actual)

    if epoch % 50 == 0:
        print(f"Epoch {epoch:4d} | L1 Loss: {total_loss.item():.4f} | Theta: {theta_deg.item():.4f}° | M: {M.item():.5f} | X: {X.item():.4f}")

print("\n--- Optimization Complete ---")
print(f"Final Extracted Theta: {theta_deg.item():.4f}")
print(f"Final Extracted M:     {M.item():.5f}")
print(f"Final Extracted X:     {X.item():.4f}")

# Save the final results to a text file for easy grading
with open("results.txt", "w") as f:
    f.write(f"Final L1 Distance Score: {total_loss.item():.4f}\n")
    f.write(f"Theta: {theta_deg.item():.4f}\n")
    f.write(f"M: {M.item():.5f}\n")
    f.write(f"X: {X.item():.4f}\n")
    
print("Results saved to results.txt" ) 
        

    
