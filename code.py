"""
Parametric Curve Optimization
Author: Kovuri Mithul
ID: AV.SC.U4AIE23025

This script uses PyTorch to perform non-linear parameter estimation.
It minimizes the L1 distance between expected (x, y) coordinates and 
a predicted parametric curve to extract three unknown variables: Theta, M, and X.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# --- 1. DATA LOADING ---
print("Loading data from xy_data.csv...")
if not os.path.exists('xy_data.csv'):
    raise FileNotFoundError("Could not find xy_data.csv. Make sure it is in the same folder as this script.")

data = pd.read_csv('xy_data.csv')

# Ensure we have the target x and y coordinates
x_expected = torch.tensor(data['x'].values, dtype=torch.float32)
y_expected = torch.tensor(data['y'].values, dtype=torch.float32)

# Check if the time parameter 't' is provided in the CSV
if 't' in data.columns:
    t_data = torch.tensor(data['t'].values, dtype=torch.float32)
else:
    # If 't' is not provided, generate uniformly sampled points from 6 to 60
    t_data = torch.tensor(np.linspace(6, 60, len(data)), dtype=torch.float32)

# --- 2. MODEL ARCHITECTURE (THE WEIGHTS) ---
# Initialize the unknown variables as PyTorch tensors that require gradients.
# We start them near the middle of their allowed bounds to help the optimizer.
theta_deg = torch.tensor([25.0], requires_grad=True) # Bound: 0 < theta < 50
M = torch.tensor([0.0], requires_grad=True)          # Bound: -0.05 < M < 0.05
X = torch.tensor([50.0], requires_grad=True)         # Bound: 0 < X < 100

# Setup the Adam optimizer to tweak our three variables
optimizer = optim.Adam([theta_deg, M, X], lr=0.1)

# The "Modulus" error function: L1 Loss (Mean Absolute Error)
criterion = nn.L1Loss()

epochs = 5000
print(f"Starting optimization for {epochs} epochs...\n")

# --- 3. TRAINING LOOP ---
for epoch in range(epochs):
    optimizer.zero_grad() # Clear gradients from the previous step
    
    # Convert theta from degrees to radians for trigonometric functions
    theta_rad = theta_deg * (np.pi / 180.0)
    
    # --- The Forward Pass ---
    # Equation 1: x = t*cos(theta) - e^(M*|t|) * sin(0.3t)*sin(theta) + X
    term_exp = torch.exp(M * torch.abs(t_data))
    term_sin_03t = torch.sin(0.3 * t_data)
    
    x_pred = (t_data * torch.cos(theta_rad)) - (term_exp * term_sin_03t * torch.sin(theta_rad)) + X
    
    # Equation 2: y = 42 + t*sin(theta) + e^(M*|t|) * sin(0.3t)*cos(theta)
    y_pred = 42.0 + (t_data * torch.sin(theta_rad)) + (term_exp * term_sin_03t * torch.cos(theta_rad))
    
    # --- Loss Calculation ---
    loss_x = criterion(x_pred, x_expected)
    loss_y = criterion(y_pred, y_expected)
    total_loss = loss_x + loss_y # Combine the L1 distances
    
    # --- Backpropagation ---
    total_loss.backward() # Calculate the gradients
    optimizer.step()      # Update the variables
    
    # --- Enforce Boundaries ---
    # We use torch.no_grad() to apply hard limits without messing up the gradient tracking
    with torch.no_grad():
        theta_deg.clamp_(0.001, 49.999)
        M.clamp_(-0.049, 0.049)
        X.clamp_(0.001, 99.999)
        
    # Log progress every 500 epochs
    if epoch % 500 == 0 or epoch == epochs - 1:
        print(f"Epoch {epoch:4d} | L1 Loss: {total_loss.item():.4f} | Theta: {theta_deg.item():.4f}° | M: {M.item():.5f} | X: {X.item():.4f}")

# --- 4. EXPORT RESULTS ---
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

print("Results saved to results.txt")

# --- 5. PLOT RESULTS ---
with torch.no_grad():
    theta_rad_final = theta_deg * (np.pi / 180.0)
    t_plot = torch.tensor(np.linspace(6, 60, 1000), dtype=torch.float32)
    term_exp_plot = torch.exp(M * torch.abs(t_plot))
    term_sin_plot = torch.sin(0.3 * t_plot)

    x_curve = (t_plot * torch.cos(theta_rad_final) 
               - term_exp_plot * term_sin_plot * torch.sin(theta_rad_final) 
               + X).numpy()
    y_curve = (42.0 
               + t_plot * torch.sin(theta_rad_final) 
               + term_exp_plot * term_sin_plot * torch.cos(theta_rad_final)).numpy()

plt.figure(figsize=(10, 7))
plt.scatter(data['x'].values, data['y'].values,
            color='steelblue', s=6, alpha=0.5, label='Data (xy_data.csv)')
plt.plot(x_curve, y_curve,
         color='crimson', linewidth=2.5, label='Fitted Curve (Adam)')

info = (f"Theta = {theta_deg.item():.4f}°\n"
        f"M     = {M.item():.5f}\n"
        f"X     = {X.item():.4f}\n"
        f"L1 Loss = {total_loss.item():.4f}")
plt.gca().text(0.05, 0.95, info, transform=plt.gca().transAxes, fontsize=10,
               verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9, edgecolor='gray'))

plt.title('Parametric Curve Fit — Adam Optimizer (code.py)', fontsize=14, fontweight='bold')
plt.xlabel('X', fontsize=12)
plt.ylabel('Y', fontsize=12)
plt.legend(fontsize=11)
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig('code_fit_plot.png', dpi=150)
print('Plot saved to code_fit_plot.png')
plt.show()