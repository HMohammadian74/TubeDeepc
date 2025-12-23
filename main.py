import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
from scipy.linalg import solve_discrete_are, pinv
from scipy.signal import chirp
import os


# ==========================================
# 1. Configuration
# ==========================================

np.random.seed(42)

DT = 0.05
N_PRED = 20
T_DATA = 600        
T_SIM = 200
X_CONSTR = 2.3       
U_CONSTR = 20.0      

# Tuning matrices
Q_MPC = np.zeros((12, 12)) 
np.fill_diagonal(Q_MPC, 0.1) 
Q_MPC[0,0] = 2000.0
Q_MPC[1,1] = 50.0

R_MPC = np.diag([0.005])
R_DELTA = np.diag([0.1])

LAMBDA_G = 1e-5
LAMBDA_SIGMA = 5e3


# ==========================================
# 2. Dynamics & Lifting
# ==========================================

def vdp_dynamics(x, u):
    if np.max(np.abs(x)) > 1e4: return np.array([np.nan, np.nan])
    u_val = u[0] if isinstance(u, (list, np.ndarray)) else u
    if hasattr(u_val, 'item'): u_val = u_val.item()
    
    def f(s, v):
        return np.array([
            2 * s[1],
            -0.5*s[0] + 2.0*s[1] - 8.0*(s[0]**2)*s[1] + v
        ])
        
    k1 = f(x, u_val)
    k2 = f(x + 0.5*DT*k1, u_val)
    k3 = f(x + 0.5*DT*k2, u_val)
    k4 = f(x + DT*k3, u_val)
    
    x_next = x + (DT/6.0)*(k1 + 2*k2 + 2*k3 + k4)
    return x_next + np.random.normal(0, 0.005, 2)

def lift_obs(x):
    x1, x2 = x[0], x[1]
    obs = np.array([
        x1, x2, 
        x1**2, x1*x2, x2**2, 
        x1**3, x2**3, 
        np.tanh(x1), np.sin(x1), x1 * np.cos(x2),
        np.exp(-0.1*x1**2), 1.0
    ])
    if len(obs) < 12: obs = np.pad(obs, (0, 12-len(obs)))
    return obs[:12]

nz = 12
nu = 1


# ==========================================
# 3. Data Collection
# ==========================================

print(">>> Collecting Data...")

def weak_controller(x):
    return -2.0 * x[0] - 2.0 * x[1] 

x_curr = np.array([0.5, 0.0])
z_raw = np.zeros((nz, T_DATA+1))
z_raw[:, 0] = lift_obs(x_curr)
u_data_list = []

t_seq = np.arange(T_DATA) * DT
chirp_sig = 10.0 * chirp(t_seq, f0=0.1, f1=2.0, t1=t_seq[-1])
random_steps = np.repeat(np.random.uniform(-5, 5, T_DATA // 20 + 1), 20)[:T_DATA]

for k in range(T_DATA):
    u_stab = weak_controller(x_curr)
    u_applied = u_stab + chirp_sig[k] + random_steps[k]
    u_applied = np.clip(u_applied, -U_CONSTR, U_CONSTR)
    
    x_next = vdp_dynamics(x_curr, [u_applied])
    x_curr = x_next
    
    z_raw[:, k+1] = lift_obs(x_next)
    u_data_list.append(u_applied)

u_data = np.array(u_data_list).reshape(1, -1)

z_mean = np.mean(z_raw, axis=1, keepdims=True)
z_std = np.std(z_raw, axis=1, keepdims=True) + 1e-6
z_data = (z_raw - z_mean) / z_std

def scale_curr(z): return (z - z_mean.flatten()) / z_std.flatten()

L = N_PRED + 1
def hankel(data):
    n, T = data.shape
    H = np.zeros((n*L, T-L+1))
    for i in range(L): H[i*n:(i+1)*n, :] = data[:, i:i + (T-L+1)]
    return H

Hz = hankel(z_data[:, :-1])
Hu = hankel(u_data)


# ==========================================
# 4. System Identification
# ==========================================

X0 = z_data[:, :-1]
X1 = z_data[:, 1:]
U0 = u_data
AB = X1 @ pinv(np.vstack([X0, U0]))
A_lin = AB[:, :nz]
B_lin = AB[:, nz:]

P_lqr = solve_discrete_are(A_lin, B_lin, Q_MPC, R_MPC)
K_LQR = np.linalg.solve(R_MPC + B_lin.T @ P_lqr @ B_lin, B_lin.T @ P_lqr @ A_lin)

# Tube calculation
errs = [np.abs(z_data[0, k+1] - (A_lin @ z_data[:,k] + B_lin @ u_data[:,k])[0]) for k in range(T_DATA)]
raw_margin = np.percentile(errs, 90) * 1.2
TUBE_MARGIN = min(raw_margin, 0.35)
X_TIGHT = (X_CONSTR - z_mean[0,0])/z_std[0,0] - TUBE_MARGIN

print(f"Tube Margin: {TUBE_MARGIN:.4f}")


# ==========================================
# 5. Controller Implementations
# ==========================================

# 5.1 Standard DeePC (Coulson et al. 2019)
def solve_standard_deepc(z_init, z_ref_traj, u_prev):
    g = cp.Variable(Hz.shape[1])
    Z = Hz @ g
    U = Hu @ g
    
    cost = 0
    constrs = [Z[:nz] == z_init, cp.abs(U) <= U_CONSTR]
    
    for k in range(N_PRED):
        idx_z = k * nz
        z_k = Z[idx_z : idx_z+nz]
        u_k = U[k*nu : (k+1)*nu]
        cost += cp.quad_form(z_k - z_ref_traj[k], Q_MPC)
        cost += cp.quad_form(u_k, R_MPC)
    
    cost += 1e-4 * cp.sum_squares(g)
    prob = cp.Problem(cp.Minimize(cost), constrs)
    
    try:
        prob.solve(solver=cp.OSQP, eps_abs=1e-3, eps_rel=1e-3, verbose=False)
        if prob.status in ['optimal', 'optimal_inaccurate']:
            return U.value[:nu]
    except:
        pass
    return -K_LQR @ (z_init - z_ref_traj[0])

# 5.2 Soft-Constrained DeePC (Elokda et al. 2021)
def solve_soft_deepc(z_init, z_ref_traj, u_prev):
    g = cp.Variable(Hz.shape[1])
    sigma = cp.Variable(nz * N_PRED)
    Z = Hz @ g
    U = Hu @ g
    
    cost = 0
    constrs = [Z[:nz] == z_init, cp.abs(U) <= U_CONSTR]
    
    limit = (X_CONSTR - z_mean[0,0])/z_std[0,0]
    for k in range(N_PRED):
        idx_z = k * nz
        z_k = Z[idx_z : idx_z+nz]
        u_k = U[k*nu : (k+1)*nu]
        sig_k = sigma[k*nz : (k+1)*nz]
        
        cost += cp.quad_form(z_k - z_ref_traj[k], Q_MPC)
        cost += cp.quad_form(u_k, R_MPC)
        constrs.append(cp.abs(z_k[0] + sig_k[0]) <= limit + 0.2)
    
    cost += 1e-4 * cp.sum_squares(g) + 1e4 * cp.norm(sigma, 1)
    prob = cp.Problem(cp.Minimize(cost), constrs)
    
    try:
        prob.solve(solver=cp.OSQP, eps_abs=1e-3, eps_rel=1e-3, verbose=False)
        if prob.status in ['optimal', 'optimal_inaccurate']:
            return U.value[:nu]
    except:
        pass
    return -K_LQR @ (z_init - z_ref_traj[0])

# 5.3 Robust DeePC (Berberich et al. 2020)
def solve_robust_deepc(z_init, z_ref_traj, u_prev):
    g = cp.Variable(Hz.shape[1])
    Z = Hz @ g
    U = Hu @ g
    
    cost = 0
    # Conservative constraint
    limit = (X_CONSTR - z_mean[0,0])/z_std[0,0] - 0.5
    constrs = [Z[:nz] == z_init, cp.abs(U) <= U_CONSTR]
    
    for k in range(N_PRED):
        idx_z = k * nz
        z_k = Z[idx_z : idx_z+nz]
        u_k = U[k*nu : (k+1)*nu]
        
        cost += cp.quad_form(z_k - z_ref_traj[k], Q_MPC)
        cost += cp.quad_form(u_k, R_MPC)
        constrs.append(cp.abs(z_k[0]) <= limit)
    
    cost += 1e-3 * cp.sum_squares(g)  # Heavy regularization
    prob = cp.Problem(cp.Minimize(cost), constrs)
    
    try:
        prob.solve(solver=cp.OSQP, eps_abs=1e-3, eps_rel=1e-3, verbose=False)
        if prob.status in ['optimal', 'optimal_inaccurate']:
            return U.value[:nu]
    except:
        pass
    return -K_LQR @ (z_init - z_ref_traj[0])

# 5.4 Tube DEEPC (Proposed)
def solve_tube_deepc(z_init, z_ref_traj, u_prev):
    g = cp.Variable(Hz.shape[1])
    sigma_y = cp.Variable(nz * N_PRED)
    
    Z = Hz @ g
    U = Hu @ g
    
    cost = 0
    constrs = [Z[:nz] == z_init, cp.abs(U) <= U_CONSTR]
    
    u_last = u_prev
    
    for k in range(N_PRED):
        idx_z = k * nz
        z_k = Z[idx_z : idx_z+nz]
        u_k = U[k*nu : (k+1)*nu]
        sig_k = sigma_y[k*nz : (k+1)*nz]
        
        ref_dist = cp.norm(z_k[0] - z_ref_traj[k][0])
        tracking_weight = 1.0 + 5.0 * cp.exp(-ref_dist)
        
        cost += tracking_weight * cp.quad_form(z_k - z_ref_traj[k], Q_MPC)
        cost += cp.quad_form(u_k, R_MPC)
        
        if k == 0: delta = u_k - u_last
        else:      delta = u_k - U[(k-1)*nu : k*nu]
        cost += cp.quad_form(delta, R_DELTA)
        
        horizon_factor = 1.0 - 0.3 * (k / N_PRED)
        slack_limit = 0.05 * (1 + k/N_PRED)
        
        constrs.append(cp.abs(z_k[0] + sig_k[0]) <= X_TIGHT * horizon_factor + slack_limit)
        constrs.append(cp.abs(sig_k[0]) <= 0.5)
    
    z_final = Z[-nz:]
    cost += 3.0 * cp.quad_form(z_final - z_ref_traj[-1], P_lqr)
    cost += LAMBDA_G * cp.sum_squares(g) + LAMBDA_SIGMA * cp.norm(sigma_y, 1)
    
    prob = cp.Problem(cp.Minimize(cost), constrs)
    try:
        prob.solve(solver=cp.OSQP, eps_abs=5e-4, eps_rel=5e-4, max_iter=8000, verbose=False)
        if prob.status in ['optimal', 'optimal_inaccurate']:
            return U.value[:nu], Z.value[:nz]
    except:
        pass
    return -K_LQR @ (z_init - z_ref_traj[0]), z_init


# ==========================================
# 6. Simulation
# ==========================================

print(">>> Running Multi-Method Comparison...")

methods = {
    "Linear MPC": {"color": "#FF6B6B", "marker": "o", "ls": "--", "alpha": 0.7, "lw": 2},
    "Standard DeePC": {"color": "#FFA500", "marker": "s", "ls": "-.", "alpha": 0.7, "lw": 2},
    "Soft-Constrained DeePC": {"color": "#9D4EDD", "marker": "^", "ls": ":", "alpha": 0.7, "lw": 2},
    "Robust DeePC": {"color": "#06D6A0", "marker": "D", "ls": "--", "alpha": 0.7, "lw": 2},
    "Tube DEEPC (Proposed)": {"color": "#1E88E5", "marker": "*", "ls": "-", "alpha": 1.0, "lw": 3.5}
}

ref_vals = np.zeros(T_SIM + N_PRED)
ref_vals[20:100] = 2.0   
ref_vals[100:160] = -2.0
ref_vals[160:] = 0.0

results = {}

for method_name in methods.keys():
    print(f"  Simulating: {method_name}")
    x = np.array([0.5, 0.0])
    hist_x = [x]
    hist_u = []
    u_prev = 0.0 
    
    for t in range(T_SIM):
        z_cur = scale_curr(lift_obs(x))
        z_refs = [scale_curr(lift_obs(np.array([ref_vals[t+k], 0]))) for k in range(N_PRED)]
        
        if method_name == "Linear MPC":
            u_apply = -K_LQR @ (z_cur - z_refs[0])
            
        elif method_name == "Standard DeePC":
            u_apply = solve_standard_deepc(z_cur, z_refs, u_prev)
            
        elif method_name == "Soft-Constrained DeePC":
            u_apply = solve_soft_deepc(z_cur, z_refs, u_prev)
            
        elif method_name == "Robust DeePC":
            u_apply = solve_robust_deepc(z_cur, z_refs, u_prev)
            
        elif method_name == "Tube DEEPC (Proposed)":
            u_prev_arr = np.array([u_prev])
            u_nom, z_nom0 = solve_tube_deepc(z_cur, z_refs, u_prev_arr)
            error_mag = np.linalg.norm(z_cur - z_nom0)
            fb_scale = min(1.5, 0.5 + error_mag)
            u_fb = -fb_scale * K_LQR @ (z_cur - z_nom0)
            u_apply = u_nom + u_fb
            
        if isinstance(u_apply, np.ndarray): u_apply = u_apply.item()
        u_apply = np.clip(u_apply, -U_CONSTR, U_CONSTR)
        
        x = vdp_dynamics(x, [u_apply])
        hist_x.append(x)
        hist_u.append(u_apply)
        u_prev = u_apply
        
    results[method_name] = {'x': np.array(hist_x), 'u': np.array(hist_u)}


# ==========================================
# 7. Performance Metrics
# ==========================================

print("\n" + "="*70)
print(" "*20 + "PERFORMANCE COMPARISON")
print("="*70)
print(f"{'Method':<30} {'MAE':>10} {'Max Error':>12} {'Violations':>12} {'Avg |u|':>10}")
print("-"*70)

for method_name in methods.keys():
    x_traj = results[method_name]['x'][:-1, 0]
    u_traj = results[method_name]['u']
    
    mae = np.mean(np.abs(x_traj - ref_vals[:T_SIM]))
    max_err = np.max(np.abs(x_traj - ref_vals[:T_SIM]))
    violations = np.sum(np.abs(x_traj) > X_CONSTR)
    avg_u = np.mean(np.abs(u_traj))
    
    print(f"{method_name:<30} {mae:>10.4f} {max_err:>12.4f} {violations:>12} {avg_u:>10.3f}")

print("="*70)


# ==========================================
# 8. Individual Plots
# ==========================================

# Create figures directory if it doesn't exist
if not os.path.exists('figures'):
    os.makedirs('figures')
    print(">>> Created 'figures' directory")

# FIGURE 1: Main Tracking Performance
fig1 = plt.figure(figsize=(14, 7))
ax = fig1.add_subplot(111)

ax.plot(ref_vals[:T_SIM], 'k--', lw=3.5, label='Reference', zorder=10)

for method_name, style in methods.items():
    x_data = results[method_name]['x'][:,0]
    ax.plot(x_data, color=style['color'], ls=style['ls'], lw=style['lw'], 
            alpha=style['alpha'], label=method_name, 
            zorder=9 if "Proposed" in method_name else 5)

ax.axhline(X_CONSTR, c='red', ls='-', lw=3, label='Safety Limit', zorder=8)
ax.axhline(-X_CONSTR, c='red', ls='-', lw=3, zorder=8)
ax.fill_between(range(T_SIM), X_CONSTR, 3.0, color='red', alpha=0.15, label='Violation Zone')
ax.fill_between(range(T_SIM), -X_CONSTR, -3.0, color='red', alpha=0.15)

ax.set_title("State Tracking Performance Comparison", fontsize=16, fontweight='bold', pad=15)
ax.set_ylabel("State $x_1$", fontsize=14)
ax.set_xlabel("Time Step", fontsize=14)
ax.legend(loc='upper right', fontsize=11, ncol=2, framealpha=0.95, shadow=True)
ax.grid(True, alpha=0.3, linestyle='--')
ax.set_ylim(-2.8, 2.8)
plt.tight_layout()
plt.savefig('figures/Fig_1_State_Tracking.png', dpi=300, bbox_inches='tight')
print(">>> Saved Fig_1_State_Tracking.png")

# FIGURE 2: Control Effort
fig2 = plt.figure(figsize=(14, 6))
ax = fig2.add_subplot(111)

for method_name, style in methods.items():
    u_data = results[method_name]['u']
    ax.plot(u_data, color=style['color'], lw=style['lw'], 
            alpha=style['alpha'], label=method_name,
            zorder=9 if "Proposed" in method_name else 5)

ax.axhline(U_CONSTR, c='red', ls=':', alpha=0.6, lw=2, label='Control Limits')
ax.axhline(-U_CONSTR, c='red', ls=':', alpha=0.6, lw=2)
ax.set_title("Control Signal Comparison", fontsize=16, fontweight='bold', pad=15)
ax.set_ylabel("Control Input $u$", fontsize=14)
ax.set_xlabel("Time Step", fontsize=14)
ax.legend(loc='upper right', fontsize=11, framealpha=0.95, shadow=True)
ax.grid(True, alpha=0.3, linestyle='--')
ax.set_xlim(0, T_SIM)
plt.tight_layout()
plt.savefig('figures/Fig_2_Control_Effort.png', dpi=300, bbox_inches='tight')
print(">>> Saved Fig_2_Control_Effort.png")

# FIGURE 3: Zoomed Rise Time Analysis
fig3 = plt.figure(figsize=(14, 7))
ax = fig3.add_subplot(111)

zoom_range = range(15, 50)
ax.plot(zoom_range, ref_vals[zoom_range], 'k--', lw=4, label='Reference', zorder=10)

for method_name, style in methods.items():
    x_data = results[method_name]['x'][zoom_range,0]
    markersize = 10 if "Proposed" in method_name else 6
    ax.plot(zoom_range, x_data, color=style['color'], marker=style['marker'], 
            markevery=3, markersize=markersize, markeredgewidth=2,
            markeredgecolor='black' if "Proposed" in method_name else 'none',
            ls=style['ls'], lw=style['lw'], alpha=style['alpha'], label=method_name,
            zorder=9 if "Proposed" in method_name else 5)

ax.axhline(X_CONSTR, c='red', ls='-', lw=2.5, label='Safety Limit')
ax.axhline(2.0, c='green', ls=':', lw=2, alpha=0.5, label='Target Value')
ax.set_title("Rise Time Comparison (Zoomed View)", fontsize=16, fontweight='bold', pad=15)
ax.set_ylabel("State $x_1$", fontsize=14)
ax.set_xlabel("Time Step", fontsize=14)
ax.legend(loc='lower right', fontsize=11, framealpha=0.95, shadow=True)
ax.grid(True, alpha=0.3, linestyle='--')
plt.tight_layout()
plt.savefig('figures/Fig_3_Rise_Time.png', dpi=300, bbox_inches='tight')
print(">>> Saved Fig_3_Rise_Time.png")

# FIGURE 4: Phase Portrait
fig4 = plt.figure(figsize=(10, 9))
ax = fig4.add_subplot(111)

for method_name, style in methods.items():
    x_data = results[method_name]['x']
    if "Proposed" in method_name:
        ax.plot(x_data[:,0], x_data[:,1], color=style['color'], lw=3.5, 
                alpha=style['alpha'], label=method_name, zorder=8)
    else:
        ax.plot(x_data[:,0], x_data[:,1], color=style['color'], lw=2, 
                alpha=style['alpha']-0.1, label=method_name, zorder=3)

ax.plot([-X_CONSTR, X_CONSTR, X_CONSTR, -X_CONSTR, -X_CONSTR], 
        [-6, -6, 6, 6, -6], 'r-', lw=3, label='Safe Region', zorder=10)
ax.scatter([0], [0], c='green', s=300, marker='*', zorder=12, 
           edgecolors='darkgreen', linewidths=3, label='Target')

ax.set_title("Phase Portrait Comparison", fontsize=16, fontweight='bold', pad=15)
ax.set_xlabel("State $x_1$", fontsize=14)
ax.set_ylabel("State $x_2$", fontsize=14)
ax.legend(loc='upper right', fontsize=11, framealpha=0.95, shadow=True)
ax.grid(True, alpha=0.3, linestyle='--')
ax.set_xlim(-3, 3)
ax.set_ylim(-7, 7)
ax.set_aspect('equal', adjustable='box')
plt.tight_layout()
plt.savefig('figures/Fig_4_Phase_Portrait.png', dpi=300, bbox_inches='tight')
print(">>> Saved Fig_4_Phase_Portrait.png")

# FIGURE 5: Tracking Error
fig5 = plt.figure(figsize=(14, 7))
ax = fig5.add_subplot(111)

for method_name, style in methods.items():
    err = np.abs(results[method_name]['x'][:-1,0] - ref_vals[:T_SIM])
    if "Proposed" in method_name:
        ax.plot(err, color=style['color'], lw=3.5, alpha=style['alpha'], 
                label=method_name, zorder=8)
        ax.fill_between(range(T_SIM), 0, err, color=style['color'], alpha=0.2, zorder=7)
    else:
        ax.plot(err, color=style['color'], lw=2, alpha=style['alpha']-0.1, 
                label=method_name, zorder=3)

ax.set_title("Absolute Tracking Error Comparison", fontsize=16, fontweight='bold', pad=15)
ax.set_ylabel("Absolute Error $|x_1 - r|$", fontsize=14)
ax.set_xlabel("Time Step", fontsize=14)
ax.legend(loc='upper right', fontsize=11, framealpha=0.95, shadow=True)
ax.grid(True, alpha=0.3, linestyle='--')
ax.set_xlim(0, T_SIM)
plt.tight_layout()
plt.savefig('figures/Fig_5_Tracking_Error.png', dpi=300, bbox_inches='tight')
print(">>> Saved Fig_5_Tracking_Error.png")

# FIGURE 6: Performance Bar Chart
fig6 = plt.figure(figsize=(12, 8))
ax = fig6.add_subplot(111)

mae_values = []
method_names_short = []

for method_name in methods.keys():
    mae = np.mean(np.abs(results[method_name]['x'][:-1,0] - ref_vals[:T_SIM]))
    mae_values.append(mae)
    short_name = method_name.replace(" (Proposed)", " ★")
    method_names_short.append(short_name)

bars = ax.barh(method_names_short, mae_values, 
               color=[methods[m]['color'] for m in methods.keys()],
               alpha=0.85, edgecolor='black', linewidth=2)

# Highlight proposed method
bars[-1].set_linewidth(4)
bars[-1].set_edgecolor('gold')

ax.set_xlabel("Mean Absolute Error (MAE)", fontsize=14, fontweight='bold')
ax.set_title("Performance Ranking by Mean Absolute Error", fontsize=16, fontweight='bold', pad=15)
ax.grid(True, alpha=0.3, axis='x', linestyle='--')

for i, v in enumerate(mae_values):
    weight = 'bold' if i == len(mae_values)-1 else 'normal'
    ax.text(v + 0.01, i, f'{v:.4f}', va='center', fontweight=weight, fontsize=12)

plt.tight_layout()
plt.savefig('figures/Fig_6_Performance_Ranking.png', dpi=300, bbox_inches='tight')
print(">>> Saved Fig_6_Performance_Ranking.png")

plt.show()

print("\n✓ All 6 individual figures generated and saved successfully!")
print("✓ Figures are located in the 'figures/' directory")
print("✓ Simulation completed successfully!")