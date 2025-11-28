import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg
import json
import csv
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader

# ==========================================
# Global Configuration
# ==========================================
HIDDEN_DIM = 64
INPUT_DIM = 10      # For Exp 1 & 2 (Matrix Analysis)
DATA_INPUT_DIM = 2  # For Exp 3-5 (Moons Dataset)
NUM_CLASSES = 2
RHO_RANGE = np.linspace(0.5, 1.5, 30)
FD_EPSILON = 1e-4
TOLERANCE = 1e-5
MAX_ITER = 100

# Set seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# ==========================================
# Model Definition
# ==========================================
class DEQLayer(nn.Module):
    def __init__(self, hidden_dim, input_dim):
        super().__init__()
        self.W = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.U = nn.Linear(input_dim, hidden_dim, bias=True)
        self.act = torch.tanh
        
        # Orthogonal init for stability analysis
        nn.init.orthogonal_(self.W.weight)
        nn.init.xavier_normal_(self.U.weight)

    def forward_step(self, z, x):
        return self.act(self.W(z) + self.U(x))

    def solve_fixed_point(self, x):
        """Simple fixed point iteration solver."""
        z = torch.zeros(x.shape[0], self.W.in_features, device=x.device)
        iterations = 0
        converged = False
        with torch.no_grad():
            for i in range(MAX_ITER):
                z_next = self.forward_step(z, x)
                if torch.norm(z_next - z) < TOLERANCE:
                    z = z_next
                    iterations = i + 1
                    converged = True
                    break
                z = z_next
                iterations = i + 1
        
        # Only do final step with gradient tracking if converged
        # This avoids moving "ahead" of the true fixed point
        if converged:
            z = self.forward_step(z, x)
        
        return z, iterations, converged

    def estimate_spectral_radius(self):
        """Power iteration estimation of rho(W)."""
        # Since we use tanh, J_f <= ||W||_2. We estimate ||W||_2.
        # For rigorous rho(J_f), we would need to account for D = diag(f').
        # Here we bound it by ||W||_2 which is standard practice.
        u = torch.randn(1, self.W.in_features).to(self.W.weight.device)
        u = u / torch.norm(u)
        with torch.no_grad():
            for _ in range(5):
                v = torch.mm(u, self.W.weight.t())
                u = torch.mm(v, self.W.weight)
                u = u / torch.norm(u)
        
        # Rayleigh quotient approximation
        v = torch.mm(u, self.W.weight.t()) 
        sigma = torch.norm(v) # Estimate of largest singular value
        return sigma

class DEQClassifier(nn.Module):
    def __init__(self, hidden_dim, input_dim, num_classes):
        super().__init__()
        self.deq = DEQLayer(hidden_dim, input_dim)
        self.classifier = nn.Linear(hidden_dim, num_classes)
        
    def forward(self, x):
        z_star, _, _ = self.deq.solve_fixed_point(x)
        return self.classifier(z_star)

# ==========================================
# Helpers
# ==========================================
def set_spectral_radius(layer, target_rho):
    """Manually set spectral radius of W for synthetic analysis."""
    with torch.no_grad():
        W_np = layer.W.weight.data.numpy()
        eigenvalues = scipy.linalg.eigvals(W_np)
        current_rho = np.max(np.abs(eigenvalues))
        scale_factor = target_rho / (current_rho + 1e-8)
        layer.W.weight.data.mul_(scale_factor)

def get_toy_data():
    """Generate non-linear Moons dataset."""
    X, y = make_moons(n_samples=1000, noise=0.1, random_state=42)
    return torch.FloatTensor(X), torch.LongTensor(y)

# ==========================================
# Experiments
# ==========================================

def run_experiment_1_and_2():
    print("\n--- Running Experiments 1 & 2: Gradient Breakdown & Solver Dynamics ---")
    model = DEQLayer(HIDDEN_DIM, INPUT_DIM)
    x = torch.randn(1, INPUT_DIM)
    
    grad_errors = []
    solver_iterations = []
    solver_converged_flags = []
    conditioning_numbers = []
    rhos = []

    for rho in RHO_RANGE:
        set_spectral_radius(model, rho)
        rhos.append(rho)
        
        # --- Experiment 2: Solver Dynamics ---
        z_star, iters, converged = model.solve_fixed_point(x)
        solver_iterations.append(iters)
        solver_converged_flags.append(converged)
        
        # --- Experiment 1: Gradient Error ---
        # 1. Analytic Gradient (Implicit)
        # J_f = D @ W
        h = model.W(z_star) + model.U(x)
        D = torch.diag((1 - torch.tanh(h)**2).squeeze())
        W = model.W.weight
        J_f = D @ W
        I = torch.eye(HIDDEN_DIM)
        
        # df/dx = D @ U
        df_dx = D @ model.U.weight
        
        try:
            term = (I - J_f)
            # Track conditioning number
            cond_num = torch.linalg.cond(term).item()
            conditioning_numbers.append(cond_num)
            
            # Solves (I-J) * v = df/dx
            grad_analytic = torch.linalg.solve(term, df_dx)
            
            # 2. Finite Difference Gradient
            grad_fd = torch.zeros(HIDDEN_DIM, INPUT_DIM)
            for i in range(INPUT_DIM):
                x_p = x.clone(); x_p[0,i] += FD_EPSILON
                x_m = x.clone(); x_m[0,i] -= FD_EPSILON
                z_p, _, _ = model.solve_fixed_point(x_p)
                z_m, _, _ = model.solve_fixed_point(x_m)
                grad_fd[:, i] = (z_p - z_m).squeeze() / (2*FD_EPSILON)
                
            diff = torch.norm(grad_analytic - grad_fd)
            rel_error = diff / (torch.norm(grad_fd) + 1e-8)
            grad_errors.append(rel_error.item())
            
        except RuntimeError:
            grad_errors.append(float('nan'))
            conditioning_numbers.append(float('nan'))

    # Plot Exp 1 & 2
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Gradients
    converged_mask = np.array(solver_converged_flags)
    ax1.scatter([r for r, c in zip(rhos, converged_mask) if c], 
                [e for e, c in zip(grad_errors, converged_mask) if c], 
                c='green', marker='o', label='Converged', alpha=0.7)
    ax1.scatter([r for r, c in zip(rhos, converged_mask) if not c], 
                [e for e, c in zip(grad_errors, converged_mask) if not c], 
                c='red', marker='x', s=100, label='Not Converged', alpha=0.7)
    ax1.set_yscale('log')
    ax1.set_xlabel(r'Spectral Radius $\Phi$')
    ax1.set_ylabel('Relative Gradient Error')
    ax1.set_title('Exp 1: Gradient Conditioning')
    ax1.axvline(1.0, color='k', linestyle='--', label='Critical Boundary')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Dynamics
    ax2.plot(rhos, solver_iterations, 'b-s')
    ax2.set_xlabel(r'Spectral Radius $\Phi$')
    ax2.set_ylabel('Solver Iterations')
    ax2.set_title('Exp 2: Solver Critical Slowing Down')
    ax2.axvline(1.0, color='k', linestyle='--', label='Critical Boundary')
    ax2.axhline(MAX_ITER, color='r', linestyle=':', label='Max Iterations')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Conditioning Number
    ax3.scatter([r for r, c in zip(rhos, converged_mask) if c], 
                [cn for cn, c in zip(conditioning_numbers, converged_mask) if c], 
                c='green', marker='o', label='Converged', alpha=0.7)
    ax3.scatter([r for r, c in zip(rhos, converged_mask) if not c], 
                [cn for cn, c in zip(conditioning_numbers, converged_mask) if not c], 
                c='red', marker='x', s=100, label='Not Converged', alpha=0.7)
    ax3.set_yscale('log')
    ax3.set_xlabel(r'Spectral Radius $\Phi$')
    ax3.set_ylabel(r'Condition Number of $(I - J_f)$')
    ax3.set_title('Matrix Conditioning vs Spectral Radius')
    ax3.axvline(1.0, color='k', linestyle='--', label='Critical Boundary')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Convergence Rate
    convergence_rate = [int(c) for c in solver_converged_flags]
    window = 3  # Rolling average window
    smoothed_rate = np.convolve(convergence_rate, np.ones(window)/window, mode='valid')
    ax4.plot(rhos[:len(smoothed_rate)], smoothed_rate, 'g-', linewidth=2)
    ax4.fill_between(rhos[:len(smoothed_rate)], 0, smoothed_rate, alpha=0.3, color='green')
    ax4.set_xlabel(r'Spectral Radius $\Phi$')
    ax4.set_ylabel('Convergence Rate (smoothed)')
    ax4.set_ylim(-0.1, 1.1)
    ax4.set_title('Solver Convergence Success Rate')
    ax4.axvline(1.0, color='k', linestyle='--', label='Critical Boundary')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('exp1_exp2_synthetic_theory.png', dpi=150)
    
    # Save data to CSV
    with open('exp1_exp2_data.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['rho', 'grad_error', 'solver_iterations', 'converged', 'conditioning_number'])
        for i in range(len(rhos)):
            writer.writerow([
                rhos[i], 
                grad_errors[i], 
                solver_iterations[i], 
                solver_converged_flags[i],
                conditioning_numbers[i]
            ])
    
    # Save summary statistics to JSON
    summary = {
        'total_runs': len(solver_converged_flags),
        'converged_runs': sum(solver_converged_flags),
        'convergence_rate': sum(solver_converged_flags) / len(solver_converged_flags),
        'critical_region_convergence': {
            'range': [0.95, 1.05],
            'converged': sum(1 for r, c in zip(rhos, solver_converged_flags) if 0.95 <= r <= 1.05 and c),
            'total': sum(1 for r in rhos if 0.95 <= r <= 1.05)
        },
        'mean_grad_error': float(np.nanmean(grad_errors)),
        'median_grad_error': float(np.nanmedian(grad_errors)),
        'mean_iterations': float(np.mean(solver_iterations)),
        'median_iterations': float(np.median(solver_iterations))
    }
    
    with open('exp1_exp2_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"Experiments 1 & 2 Completed. Plot saved.")
    print(f"  Convergence rate: {sum(solver_converged_flags)}/{len(solver_converged_flags)} runs converged")
    print(f"  Critical region (ρ ∈ [0.95, 1.05]): {sum(1 for r, c in zip(rhos, solver_converged_flags) if 0.95 <= r <= 1.05 and c)}/{sum(1 for r in rhos if 0.95 <= r <= 1.05)} converged")
    print(f"  Data saved to: exp1_exp2_data.csv, exp1_exp2_summary.json")


def run_experiment_3_expressivity():
    print("\n--- Running Experiment 3: Expressivity Peak (Phi-Regularization) ---")
    X, y = get_toy_data()
    
    # Train/test split for generalization measurement
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    train_ds = TensorDataset(X_train, y_train)
    train_dl = DataLoader(train_ds, batch_size=32, shuffle=True)
    
    # Targets for regularization: Over-stable, Critical, Unstable
    targets = [0.5, 0.95, 1.2]
    final_train_accuracies = []
    final_test_accuracies = []
    final_phi_values = []
    
    for tau in targets:
        print(f"Training with target Phi = {tau}...")
        model = DEQClassifier(HIDDEN_DIM, DATA_INPUT_DIM, NUM_CLASSES)
        opt = optim.Adam(model.parameters(), lr=0.01)
        criterion = nn.CrossEntropyLoss()
        
        # Training Loop
        for epoch in range(15): # Fast training
            for bx, by in train_dl:
                opt.zero_grad()
                logits = model(bx)
                
                # Phi Regularization Term
                # L_total = L_class + lambda * (phi - tau)^2
                phi_est = model.deq.estimate_spectral_radius()
                # Two-sided penalty to force it TO the target regime
                reg_loss = 0.5 * (phi_est - tau)**2 
                
                loss = criterion(logits, by) + reg_loss
                loss.backward()
                opt.step()
        
        # Eval on train and test
        with torch.no_grad():
            # Train accuracy
            train_logits = model(X_train)
            train_preds = torch.argmax(train_logits, dim=1)
            train_acc = (train_preds == y_train).float().mean().item()
            final_train_accuracies.append(train_acc)
            
            # Test accuracy (generalization)
            test_logits = model(X_test)
            test_preds = torch.argmax(test_logits, dim=1)
            test_acc = (test_preds == y_test).float().mean().item()
            final_test_accuracies.append(test_acc)
            
            # Final learned phi
            phi_final = model.deq.estimate_spectral_radius().item()
            final_phi_values.append(phi_final)
            
            print(f"  -> Train Accuracy: {train_acc*100:.2f}%, Test Accuracy: {test_acc*100:.2f}%, Final Phi: {phi_final:.3f}")

    # Plot Exp 3
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    x_pos = np.arange(len(targets))
    width = 0.35
    
    # Plot 1: Accuracies
    ax1.bar(x_pos - width/2, final_train_accuracies, width, label='Train', color='skyblue')
    ax1.bar(x_pos + width/2, final_test_accuracies, width, label='Test', color='orange')
    ax1.set_xlabel(r'Regularization Target $\tau$')
    ax1.set_ylabel('Accuracy')
    ax1.set_ylim(0.5, 1.0)
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels([f'{t}' for t in targets])
    ax1.set_title('Exp 3: Expressivity Peak at Criticality')
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Plot 2: Learned Phi vs Target
    ax2.scatter(targets, final_phi_values, s=100, c='purple', alpha=0.6, edgecolors='black', linewidth=2)
    ax2.plot([min(targets), max(targets)], [min(targets), max(targets)], 'k--', label='Target Line', alpha=0.5)
    ax2.set_xlabel(r'Target $\tau$')
    ax2.set_ylabel(r'Learned $\Phi$')
    ax2.set_title('Learned Spectral Radius vs Target')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('exp3_expressivity.png', dpi=150)
    
    # Save data to CSV
    with open('exp3_expressivity_data.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['target_tau', 'train_accuracy', 'test_accuracy', 'final_phi'])
        for i in range(len(targets)):
            writer.writerow([
                targets[i],
                final_train_accuracies[i],
                final_test_accuracies[i],
                final_phi_values[i]
            ])
    
    # Save summary to JSON
    summary = {
        'experiment': 'expressivity_peak',
        'description': 'Comparing model performance across different spectral radius targets',
        'results': [
            {
                'target_tau': float(targets[i]),
                'train_accuracy': float(final_train_accuracies[i]),
                'test_accuracy': float(final_test_accuracies[i]),
                'generalization_gap': float(final_train_accuracies[i] - final_test_accuracies[i]),
                'final_phi': float(final_phi_values[i]),
                'phi_error': float(abs(final_phi_values[i] - targets[i]))
            }
            for i in range(len(targets))
        ],
        'best_test_accuracy': {
            'tau': float(targets[np.argmax(final_test_accuracies)]),
            'accuracy': float(max(final_test_accuracies))
        }
    }
    
    with open('exp3_expressivity_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print("Experiment 3 Completed. Plot saved.")
    print(f"  Data saved to: exp3_expressivity_data.csv, exp3_expressivity_summary.json")


def run_experiment_4_weight_norm():
    print("\n--- Running Experiment 4: Phi-Reg vs Weight Norm ---")
    X, y = get_toy_data()
    dl = DataLoader(TensorDataset(X, y), batch_size=32, shuffle=True)
    
    # 1. Phi Reg Model (Target = 0.95)
    model_phi = DEQClassifier(HIDDEN_DIM, DATA_INPUT_DIM, NUM_CLASSES)
    opt_phi = optim.Adam(model_phi.parameters(), lr=0.01)
    
    # 2. Weight Norm Model (Strict constraint ||W|| < 1)
    model_wn = DEQClassifier(HIDDEN_DIM, DATA_INPUT_DIM, NUM_CLASSES)
    # Apply weight norm hook
    model_wn.deq.W = nn.utils.weight_norm(model_wn.deq.W, name='weight')
    opt_wn = optim.Adam(model_wn.parameters(), lr=0.01)
    
    loss_phi_curve = []
    loss_wn_curve = []
    
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(20):
        # Train Phi
        epoch_loss_phi = 0
        for bx, by in dl:
            opt_phi.zero_grad()
            logits = model_phi(bx)
            phi = model_phi.deq.estimate_spectral_radius()
            reg = 0.1 * torch.relu(phi - 0.95)**2
            loss = criterion(logits, by) + reg
            loss.backward()
            opt_phi.step()
            epoch_loss_phi += loss.item()
        loss_phi_curve.append(epoch_loss_phi / len(dl))
        
        # Train WN
        epoch_loss_wn = 0
        for bx, by in dl:
            opt_wn.zero_grad()
            # Force WN constraint roughly by penalizing norm > 0.95
            # Or assume standard weight_norm param maintains it if initialized well.
            # Here we add explicit penalty on weights to simulate strict WN constraint
            logits = model_wn(bx)
            w_norm = torch.norm(model_wn.deq.W.weight)
            reg = 1.0 * torch.relu(w_norm - 0.95)**2 
            loss = criterion(logits, by) + reg
            loss.backward()
            opt_wn.step()
            epoch_loss_wn += loss.item()
        loss_wn_curve.append(epoch_loss_wn / len(dl))

    # Plot Exp 4
    plt.figure(figsize=(6, 5))
    plt.plot(loss_phi_curve, label='Phi-Regularization')
    plt.plot(loss_wn_curve, label='Weight Normalization', linestyle='--')
    plt.xlabel('Epochs')
    plt.ylabel('Training Loss')
    plt.title('Exp 4: Convergence Speed')
    plt.legend()
    plt.savefig('exp4_weight_norm.png')
    
    # Save data to CSV
    with open('exp4_weight_norm_data.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'phi_reg_loss', 'weight_norm_loss'])
        for i in range(len(loss_phi_curve)):
            writer.writerow([i+1, loss_phi_curve[i], loss_wn_curve[i]])
    
    # Save summary to JSON
    summary = {
        'experiment': 'phi_reg_vs_weight_norm',
        'epochs': len(loss_phi_curve),
        'phi_regularization': {
            'final_loss': float(loss_phi_curve[-1]),
            'min_loss': float(min(loss_phi_curve)),
            'convergence_speed': 'fast' if loss_phi_curve[-1] < loss_wn_curve[-1] else 'slow'
        },
        'weight_normalization': {
            'final_loss': float(loss_wn_curve[-1]),
            'min_loss': float(min(loss_wn_curve)),
        },
        'winner': 'phi_reg' if loss_phi_curve[-1] < loss_wn_curve[-1] else 'weight_norm'
    }
    
    with open('exp4_weight_norm_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print("Experiment 4 Completed. Plot saved.")
    print(f"  Data saved to: exp4_weight_norm_data.csv, exp4_weight_norm_summary.json")


def run_experiment_5_meta_stability():
    print("\n--- Running Experiment 5: Meta-Learned Stability ---")
    # Simulate a meta-controller that adjusts lambda based on Phi
    # Heuristic: If Phi > 1.0, spike lambda. If Phi < 0.9, lower lambda to allow exploration.
    
    X, y = get_toy_data()
    dl = DataLoader(TensorDataset(X, y), batch_size=32, shuffle=True)
    
    model = DEQClassifier(HIDDEN_DIM, DATA_INPUT_DIM, NUM_CLASSES)
    opt = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    
    phi_history = []
    lambda_history = []
    
    current_lambda = 0.1
    
    for epoch in range(20):
        avg_phi = 0
        for bx, by in dl:
            opt.zero_grad()
            logits = model(bx)
            
            # --- Meta Controller Logic ---
            phi = model.deq.estimate_spectral_radius()
            if phi > 1.0:
                current_lambda = min(current_lambda * 1.5, 10.0) # Emergency brake
            elif phi < 0.85:
                current_lambda = max(current_lambda * 0.9, 0.01) # Encourage growth
            # -----------------------------
            
            reg = current_lambda * torch.relu(phi - 0.95)**2
            loss = criterion(logits, by) + reg
            loss.backward()
            opt.step()
            
            avg_phi += phi.item()
            
        phi_history.append(avg_phi / len(dl))
        lambda_history.append(current_lambda)

    # Plot Exp 5
    fig, ax1 = plt.subplots(figsize=(8, 5))
    
    color = 'tab:blue'
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel(r'Spectral Radius $\Phi$', color=color)
    ax1.plot(phi_history, color=color, label='Phi')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.axhline(1.0, color='k', linestyle=':', alpha=0.5)
    
    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel(r'Meta-Learned $\lambda$', color=color)
    ax2.plot(lambda_history, color=color, linestyle='--', label='Lambda')
    ax2.tick_params(axis='y', labelcolor=color)
    
    plt.title('Exp 5: Meta-Controller Dynamics')
    fig.tight_layout()
    plt.savefig('exp5_meta_stability.png')
    
    # Save data to CSV
    with open('exp5_meta_stability_data.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'phi', 'lambda'])
        for i in range(len(phi_history)):
            writer.writerow([i+1, phi_history[i], lambda_history[i]])
    
    # Save summary to JSON
    summary = {
        'experiment': 'meta_learned_stability',
        'target_phi': 0.95,
        'epochs': len(phi_history),
        'phi_statistics': {
            'min': float(min(phi_history)),
            'max': float(max(phi_history)),
            'mean': float(np.mean(phi_history)),
            'final': float(phi_history[-1]),
            'std': float(np.std(phi_history))
        },
        'lambda_statistics': {
            'min': float(min(lambda_history)),
            'max': float(max(lambda_history)),
            'mean': float(np.mean(lambda_history)),
            'final': float(lambda_history[-1])
        },
        'convergence': {
            'final_error': float(abs(phi_history[-1] - 0.95)),
            'stayed_subcritical': all(p < 1.2 for p in phi_history),
            'oscillations': int(sum(1 for i in range(1, len(phi_history)) if (phi_history[i] - phi_history[i-1]) * (phi_history[i-1] - phi_history[max(0, i-2)]) < 0))
        }
    }
    
    with open('exp5_meta_stability_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print("Experiment 5 Completed. Plot saved.")
    print(f"  Data saved to: exp5_meta_stability_data.csv, exp5_meta_stability_summary.json")


# ==========================================
# Unit Tests (pytest-compatible)
# ==========================================

def test_gradient_correctness_well_conditioned():
    """Test 1: Gradient correctness in a well-conditioned regime (rho = 0.5)."""
    print("\n--- Unit Test 1: Gradient Correctness (Well-Conditioned) ---")
    
    hidden_dim = 8  # Small for speed
    input_dim = 5
    model = DEQLayer(hidden_dim, input_dim)
    set_spectral_radius(model, 0.5)
    
    x = torch.randn(1, input_dim)
    z_star, iters, converged = model.solve_fixed_point(x)
    
    assert converged, "Fixed point should converge for rho=0.5"
    
    # Analytic gradient
    h = model.W(z_star) + model.U(x)
    D = torch.diag((1 - torch.tanh(h)**2).squeeze())
    W = model.W.weight
    J_f = D @ W
    I = torch.eye(hidden_dim)
    df_dx = D @ model.U.weight
    
    grad_analytic = torch.linalg.solve(I - J_f, df_dx)
    
    # Finite difference gradient
    grad_fd = torch.zeros(hidden_dim, input_dim)
    for i in range(input_dim):
        x_p = x.clone(); x_p[0,i] += FD_EPSILON
        x_m = x.clone(); x_m[0,i] -= FD_EPSILON
        z_p, _, _ = model.solve_fixed_point(x_p)
        z_m, _, _ = model.solve_fixed_point(x_m)
        grad_fd[:, i] = (z_p - z_m).squeeze() / (2*FD_EPSILON)
    
    rel_error = torch.norm(grad_analytic - grad_fd) / (torch.norm(grad_fd) + 1e-8)
    
    print(f"  Relative Error: {rel_error.item():.6f}")
    assert rel_error < 1e-2, f"Gradient error should be small, got {rel_error.item()}"
    print("  ✅ PASSED")


def test_gradient_blowup_near_critical():
    """Test 2: Gradient blow-up near rho ≈ 1 (ill-conditioned)."""
    print("\n--- Unit Test 2: Gradient Blow-up (Critical Regime) ---")
    
    hidden_dim = 8
    input_dim = 5
    model = DEQLayer(hidden_dim, input_dim)
    set_spectral_radius(model, 0.99)
    
    x = torch.randn(1, input_dim)
    z_star, iters, converged = model.solve_fixed_point(x)
    
    # Build Jacobian
    h = model.W(z_star) + model.U(x)
    D = torch.diag((1 - torch.tanh(h)**2).squeeze())
    W = model.W.weight
    J_f = D @ W
    I = torch.eye(hidden_dim)
    
    cond_num = torch.linalg.cond(I - J_f).item()
    
    print(f"  Condition Number: {cond_num:.2e}")
    print(f"  Converged: {converged}")
    print(f"  Iterations: {iters}")
    
    # Expect higher condition number than well-conditioned case (rho=0.5)
    # or more iterations showing critical slowing down
    assert cond_num > 2.0 or iters > 20, f"System should show signs of critical behavior, got cond={cond_num}, iters={iters}"
    print("  ✅ PASSED")


def test_solver_iteration_monotonicity():
    """Test 3: Solver iterations increase with rho (critical slowing down)."""
    print("\n--- Unit Test 3: Solver Iteration Monotonicity ---")
    
    hidden_dim = 16
    input_dim = 8
    model = DEQLayer(hidden_dim, input_dim)
    x = torch.randn(1, input_dim)
    
    rho_values = [0.5, 0.7, 0.9, 1.1]
    iterations_list = []
    
    for rho in rho_values:
        set_spectral_radius(model, rho)
        _, iters, _ = model.solve_fixed_point(x)
        iterations_list.append(iters)
        print(f"  rho={rho}: {iters} iterations")
    
    # Check monotonicity up to critical point
    assert iterations_list[1] > iterations_list[0], "Iterations should increase with rho"
    assert iterations_list[2] > iterations_list[1], "Iterations should increase approaching critical point"
    # For supercritical, just check it takes significantly more than subcritical
    assert iterations_list[3] > iterations_list[0] * 1.5, f"Supercritical should take much longer than subcritical, got {iterations_list[3]} vs {iterations_list[0]}"
    
    print("  ✅ PASSED")


def test_meta_controller_convergence():
    """Test 4: Meta-controller moves phi toward target."""
    print("\n--- Unit Test 4: Meta-Controller Convergence ---")
    
    X, y = get_toy_data()
    dl = DataLoader(TensorDataset(X, y), batch_size=32, shuffle=True)
    
    model = DEQClassifier(HIDDEN_DIM, DATA_INPUT_DIM, NUM_CLASSES)
    opt = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    
    phi_history = []
    current_lambda = 0.1
    target_phi = 0.95
    
    # Reduced epochs for speed
    for epoch in range(10):
        avg_phi = 0
        for bx, by in dl:
            opt.zero_grad()
            logits = model(bx)
            
            # Meta Controller Logic
            phi = model.deq.estimate_spectral_radius()
            if phi > 1.0:
                current_lambda = min(current_lambda * 1.5, 10.0)
            elif phi < 0.85:
                current_lambda = max(current_lambda * 0.9, 0.01)
            
            reg = current_lambda * torch.relu(phi - target_phi)**2
            loss = criterion(logits, by) + reg
            loss.backward()
            opt.step()
            
            avg_phi += phi.item()
            
        phi_history.append(avg_phi / len(dl))
    
    print(f"  Phi range: [{min(phi_history):.3f}, {max(phi_history):.3f}]")
    print(f"  Final phi: {phi_history[-1]:.3f}")
    
    # Check that phi oscillates near target and converges close to it
    assert min(phi_history) < 1.0, "Phi should stay subcritical at some point"
    assert max(phi_history) > 0.85, "Phi should explore near target"
    assert abs(phi_history[-1] - target_phi) < 0.15, f"Final phi should be near target, got {phi_history[-1]}"
    
    print("  ✅ PASSED")


def run_all_unit_tests():
    """Run all unit tests."""
    print("\n" + "="*60)
    print("Running Unit Tests")
    print("="*60)
    
    test_results = []
    
    # Test 1
    try:
        test_gradient_correctness_well_conditioned()
        test_results.append({'test': 'gradient_correctness_well_conditioned', 'status': 'PASSED'})
    except AssertionError as e:
        test_results.append({'test': 'gradient_correctness_well_conditioned', 'status': 'FAILED', 'error': str(e)})
    
    # Test 2
    try:
        test_gradient_blowup_near_critical()
        test_results.append({'test': 'gradient_blowup_near_critical', 'status': 'PASSED'})
    except AssertionError as e:
        test_results.append({'test': 'gradient_blowup_near_critical', 'status': 'FAILED', 'error': str(e)})
    
    # Test 3
    try:
        test_solver_iteration_monotonicity()
        test_results.append({'test': 'solver_iteration_monotonicity', 'status': 'PASSED'})
    except AssertionError as e:
        test_results.append({'test': 'solver_iteration_monotonicity', 'status': 'FAILED', 'error': str(e)})
    
    # Test 4
    try:
        test_meta_controller_convergence()
        test_results.append({'test': 'meta_controller_convergence', 'status': 'PASSED'})
    except AssertionError as e:
        test_results.append({'test': 'meta_controller_convergence', 'status': 'FAILED', 'error': str(e)})
    
    # Save test results
    summary = {
        'total_tests': len(test_results),
        'passed': sum(1 for r in test_results if r['status'] == 'PASSED'),
        'failed': sum(1 for r in test_results if r['status'] == 'FAILED'),
        'results': test_results
    }
    
    with open('unit_tests_results.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print("\n" + "="*60)
    if all(r['status'] == 'PASSED' for r in test_results):
        print("All Unit Tests Passed! ✅")
    else:
        print(f"Some tests failed: {summary['passed']}/{summary['total_tests']} passed")
    print("="*60)
    print("  Test results saved to: unit_tests_results.json")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == '--unit-tests':
        # Run only unit tests
        run_all_unit_tests()
    elif len(sys.argv) > 1 and sys.argv[1] == '--all':
        # Run experiments then unit tests
        run_experiment_1_and_2()
        run_experiment_3_expressivity()
        run_experiment_4_weight_norm()
        run_experiment_5_meta_stability()
        print("\nAll experiments completed successfully.")
        run_all_unit_tests()
    else:
        # Default: run experiments only
        run_experiment_1_and_2()
        run_experiment_3_expressivity()
        run_experiment_4_weight_norm()
        run_experiment_5_meta_stability()
        print("\nAll experiments completed successfully.")
        print("\nTip: Run with '--unit-tests' for unit tests only, or '--all' for both.")