import os
import numpy as np
import scipy.io
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# Set random seeds
torch.manual_seed(42)
np.random.seed(42)

# 1. Load Data
mat_path = r"C:\Users\kevor\OneDrive\Desktop\Lucy_AI Training\hw_1\task 5\SilverboxFiles\SNLS80mV.mat"
data = scipy.io.loadmat(mat_path)

# V1: input voltage u(t), V2: output voltage y(t)
u_raw = data['V1'].squeeze()
y_raw = data['V2'].squeeze()

fs = 610.35
dt = 1.0 / fs

# Excerpt: slice [42650:46746] (4096 samples)
u_excerpt = u_raw[42650:46746].copy()
y_excerpt = y_raw[42650:46746].copy()

# Centering
u_excerpt -= np.mean(u_excerpt)
y_excerpt -= np.mean(y_excerpt)

N_total = len(y_excerpt) # 4096
N_train = 3072
N_test = 1024

# Compute velocities using central finite differences for initial states
v_excerpt = np.zeros_like(y_excerpt)
v_excerpt[1:-1] = (y_excerpt[2:] - y_excerpt[:-2]) / (2 * dt)
v_excerpt[0] = (y_excerpt[1] - y_excerpt[0]) / dt
v_excerpt[-1] = (y_excerpt[-1] - y_excerpt[-2]) / dt

# Convert to tensors
u_tensor = torch.tensor(u_excerpt, dtype=torch.float32)
y_tensor = torch.tensor(y_excerpt, dtype=torch.float32)
v_tensor = torch.tensor(v_excerpt, dtype=torch.float32)

# 2. Model Definition
class AccelerationNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 64),
            nn.SiLU(),
            nn.Linear(64, 64),
            nn.SiLU(),
            nn.Linear(64, 64),
            nn.SiLU(),
            nn.Linear(64, 1)
        )
        
    def forward(self, y, v, u):
        # Scale inputs
        y_scaled = y / 0.05
        v_scaled = v / 20.0
        u_scaled = u / 0.05
        inp = torch.stack([y_scaled, v_scaled, u_scaled], dim=-1)
        out = self.net(inp).squeeze(-1)
        # Scale acceleration output
        return out * 1000.0

model = AccelerationNet()

def rk4_step(model, y, v, u_curr, u_next, dt):
    u_mid = 0.5 * (u_curr + u_next)
    
    # k1
    a1 = model(y, v, u_curr)
    dy1 = v
    dv1 = a1
    
    # k2
    y2 = y + 0.5 * dt * dy1
    v2 = v + 0.5 * dt * dv1
    a2 = model(y2, v2, u_mid)
    dy2 = v2
    dv2 = a2
    
    # k3
    y3 = y + 0.5 * dt * dy2
    v3 = v + 0.5 * dt * dv2
    a3 = model(y3, v3, u_mid)
    dy3 = v3
    dv3 = a3
    
    # k4
    y4 = y + dt * dy3
    v4 = v + dt * dv3
    a4 = model(y4, v4, u_next)
    dy4 = v4
    dv4 = a4
    
    y_next = y + (dt / 6.0) * (dy1 + 2.0 * dy2 + 2.0 * dy3 + dy4)
    v_next = v + (dt / 6.0) * (dv1 + 2.0 * dv2 + 2.0 * dv3 + dv4)
    
    return y_next, v_next

# 3. Training
L = 64
batch_size = 32
num_steps = 3500

optimizer = optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-5)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_steps, eta_min=1e-5)

max_start_idx = N_train - L

for step in range(num_steps):
    optimizer.zero_grad()
    
    # Random batch start indices
    start_indices = torch.randint(0, max_start_idx, (batch_size,))
    
    # Initial states for batch
    y_curr = y_tensor[start_indices]
    v_curr = v_tensor[start_indices]
    
    y_preds = []
    
    for t in range(L):
        u_curr = u_tensor[start_indices + t]
        u_next = u_tensor[start_indices + t + 1]
        
        y_next, v_next = rk4_step(model, y_curr, v_curr, u_curr, u_next, dt)
        y_preds.append(y_next)
        
        y_curr = y_next
        v_curr = v_next
        
    y_preds = torch.stack(y_preds, dim=1) # [batch_size, L]
    
    # Targets
    target_indices = start_indices.unsqueeze(1) + torch.arange(1, L + 1).unsqueeze(0)
    y_targets = y_tensor[target_indices]
    
    loss = torch.mean((y_preds - y_targets) ** 2)
    loss.backward()
    
    # Gradient clipping for stability
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    
    optimizer.step()
    scheduler.step()

# 4. Free-Run Evaluation
model.eval()
with torch.no_grad():
    y_sim = torch.zeros(N_total)
    v_sim = torch.zeros(N_total)
    
    # Initial condition at start of full sequence
    y_sim[0] = y_tensor[0]
    v_sim[0] = v_tensor[0]
    
    for k in range(N_total - 1):
        u_curr = u_tensor[k]
        u_next = u_tensor[k + 1]
        y_next, v_next = rk4_step(model, y_sim[k], v_sim[k], u_curr, u_next, dt)
        y_sim[k + 1] = y_next
        v_sim[k + 1] = v_next

y_sim_np = y_sim.numpy()
y_true_np = y_excerpt

train_rmse = np.sqrt(np.mean((y_sim_np[:N_train] - y_true_np[:N_train])**2))
test_rmse = np.sqrt(np.mean((y_sim_np[N_train:] - y_true_np[N_train:])**2))

print(f"TRAIN_RMSE: {train_rmse:.6f}")
print(f"TEST_RMSE: {test_rmse:.6f}")

# 5. Plotting
plt.figure(figsize=(10, 4))
t_test = np.arange(N_train, N_total) * dt
plt.plot(t_test, y_true_np[N_train:], label='Measured y(t)', color='black', alpha=0.8)
plt.plot(t_test, y_sim_np[N_train:], label='Predicted y(t)', color='red', linestyle='--')
plt.xlabel('Time (s)')
plt.ylabel('Output Voltage (V)')
plt.title('Silverbox Test Free Run (Last 1024 samples)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('test_free_run.png', dpi=150)
