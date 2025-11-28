import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_default_dtype(torch.float32)


# ============================================
# 1. Heterogeneous PDE data generation
#    Solve ∇·(k∇u)=0 with random k(x,y)
# ============================================

def generate_random_boundary(batch_size, H, W, device=device):
    """
    Random Dirichlet boundary conditions on an HxW grid.
    boundary: [B,1,H,W], nonzero only on edges.
    mask    : [B,1,H,W], 1 on edges, 0 in interior.
    """
    boundary = torch.zeros(batch_size, 1, H, W, device=device)
    mask = torch.zeros_like(boundary)

    # random edge values in [-1,1]
    top = torch.rand(batch_size, 1, W, device=device) * 2 - 1
    bottom = torch.rand(batch_size, 1, W, device=device) * 2 - 1
    left = torch.rand(batch_size, 1, H, device=device) * 2 - 1
    right = torch.rand(batch_size, 1, H, device=device) * 2 - 1

    boundary[:, :, 0, :] = top
    boundary[:, :, -1, :] = bottom
    boundary[:, :, :, 0] = left
    boundary[:, :, :, -1] = right

    mask[:, :, 0, :] = 1.0
    mask[:, :, -1, :] = 1.0
    mask[:, :, :, 0] = 1.0
    mask[:, :, :, -1] = 1.0

    return boundary, mask


def generate_random_k(batch_size, H, W, device=device):
    """
    Random heterogeneous conductivity k(x,y) in [k_min, k_max], smoothed a bit.
    k: [B,1,H,W]
    """
    k_min, k_max = 0.1, 3.0
    log_k = torch.randn(batch_size, 1, H, W, device=device)

    # simple 3x3 average blur to make k spatially smooth
    kernel = torch.ones(1, 1, 3, 3, device=device) / 9.0
    log_k = F.conv2d(log_k, kernel, padding=1)
    log_k = torch.clamp(log_k, -1.0, 1.0)

    k = torch.exp(log_k)
    k = (k - k.min()) / (k.max() - k.min() + 1e-8)
    k = k_min + (k_max - k_min) * k
    return k


def hetero_step(u, k, boundary, mask):
    """
    One Jacobi-like iteration for ∇·(k∇u)=0:
    approximate as weighted neighbor average with weights from k at neighbors.

    u, k, boundary, mask: [B,1,H,W]
    """
    B, C, H, W = u.shape
    # Pad by 1 on all sides: [B, 1, H, W] -> [B, 1, H+2, W+2]
    u_pad = F.pad(u, (1, 1, 1, 1), mode="replicate")
    k_pad = F.pad(k, (1, 1, 1, 1), mode="replicate")

    # Extract 4-neighbors for each interior point
    # Interior points in original grid: [1:H-1, 1:W-1]
    # In padded grid they're at [2:H, 2:W]
    # North: one row up -> [1:H-1, 2:W]
    # South: one row down -> [3:H+1, 2:W]
    # West: one col left -> [2:H, 1:W-1]
    # East: one col right -> [2:H, 3:W+1]
    
    u_n = u_pad[:, :, 1:-2, 2:-1]  # [B, 1, H-1, W-1]
    u_s = u_pad[:, :, 3:,   2:-1]  # [B, 1, H-1, W-1]
    u_w = u_pad[:, :, 2:-1, 1:-2]  # [B, 1, H-1, W-1]
    u_e = u_pad[:, :, 2:-1, 3:  ]  # [B, 1, H-1, W-1]

    k_n = k_pad[:, :, 1:-2, 2:-1]
    k_s = k_pad[:, :, 3:,   2:-1]
    k_w = k_pad[:, :, 2:-1, 1:-2]
    k_e = k_pad[:, :, 2:-1, 3:  ]

    # Wait, this gives [H-1, W-1] but we need [H-2, W-2] for u[:,:,1:-1,1:-1]
    # Let me rethink: interior is u[:,:,1:H-1, 1:W-1] which has shape [H-2, W-2]
    # In padded coords this is u_pad[:,:, 2:H, 2:W]
    # North neighbors: u_pad[:,:, 1:H-1, 2:W]
    # South neighbors: u_pad[:,:, 3:H+1, 2:W] 
    # West neighbors: u_pad[:,:, 2:H, 1:W-1]
    # East neighbors: u_pad[:,:, 2:H, 3:W+1]
    
    u_n = u_pad[:, :, 1:H-1, 2:W]
    u_s = u_pad[:, :, 3:H+1, 2:W]
    u_w = u_pad[:, :, 2:H, 1:W-1]
    u_e = u_pad[:, :, 2:H, 3:W+1]

    k_n = k_pad[:, :, 1:H-1, 2:W]
    k_s = k_pad[:, :, 3:H+1, 2:W]
    k_w = k_pad[:, :, 2:H, 1:W-1]
    k_e = k_pad[:, :, 2:H, 3:W+1]

    num = k_n * u_n + k_s * u_s + k_w * u_w + k_e * u_e
    den = k_n + k_s + k_w + k_e + 1e-6

    u_new = u.clone()
    u_new[:, :, 1:-1, 1:-1] = num / den

    # enforce boundary
    u_new = u_new * (1.0 - mask) + boundary * mask
    return u_new


def solve_hetero_pde(boundary, mask, k, n_iter=800):
    """
    Jacobi-like relaxation to solve ∇·(k∇u)=0 with Dirichlet boundary.
    """
    u = boundary.clone()
    for _ in range(n_iter):
        u = hetero_step(u, k, boundary, mask)
    return u


def make_hetero_pde_dataset(n_samples, H, W, n_iter=800, device=device, seed=0):
    """
    Generate (input, solution) pairs for heterogeneous PDE.
    Input: [boundary, mask, k] as 3 channels.
    Target: full solution u.
    """
    g = torch.Generator(device=device).manual_seed(seed)
    batch_size = 64
    xs, ys = [], []
    n_done = 0

    while n_done < n_samples:
        bs = min(batch_size, n_samples - n_done)
        boundary, mask = generate_random_boundary(bs, H, W, device=device)
        k = generate_random_k(bs, H, W, device=device)
        with torch.no_grad():
            u = solve_hetero_pde(boundary, mask, k, n_iter=n_iter)
        inp = torch.cat([boundary, mask, k], dim=1)
        xs.append(inp)
        ys.append(u)
        n_done += bs

    X = torch.cat(xs, dim=0)
    Y = torch.cat(ys, dim=0)
    return TensorDataset(X, Y)


# ============================================
# 2. Three-network DEQ PDE block (heterogeneous)
#    - core solver f_theta
#    - stabilizer g_phi (per-pixel alpha)
#    - spectral controller h_psi (global gamma)
# ============================================

class StabilizerNet(nn.Module):
    """
    Local stabilizer: produces per-pixel relaxation alpha(x) ∈ (0, 1).
    Input: [boundary, mask, k].
    """
    def __init__(self, in_ch=3, hidden_ch=16):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, hidden_ch, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(hidden_ch, 1, kernel_size=3, padding=1)

    def forward(self, boundary, mask, k):
        x = torch.cat([boundary, mask, k], dim=1)
        h = F.gelu(self.conv1(x))
        alpha = torch.sigmoid(self.conv2(h))  # (0,1)
        return alpha


class SpectralController(nn.Module):
    """
    Global spectral controller with Physics-Informed Scaling.
    
    Instead of asking the network to learn the CFL condition (dt ~ dx^2),
    we enforce it structurally via hard-coded scaling.
    
    The network learns material-dependent stiffness (base_gamma),
    but physics enforces the resolution scaling: gamma_final = base_gamma * (32/H)^2
    """
    def __init__(self, in_dim=5, hidden=32):
        # Back to in_dim=5 (removed the 1/H feature since we hard-code it)
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.fc2 = nn.Linear(hidden, 1)

    def forward(self, boundary, mask, k):
        """
        Hybrid Intelligence: Known Physics + Learned Adaptation
        
        Features:
        - mean |boundary|
        - std |boundary|
        - mean k
        - std k
        - constant 1
        """
        B = boundary.size(0)
        _, _, H, W = boundary.shape
        
        # 1. Compute Features (Same as before)
        b_abs = boundary.abs()
        mean_b = b_abs.view(B, -1).mean(dim=1, keepdim=True)
        std_b = b_abs.view(B, -1).std(dim=1, keepdim=True)

        mean_k = k.view(B, -1).mean(dim=1, keepdim=True)
        std_k = k.view(B, -1).std(dim=1, keepdim=True)

        ones = torch.ones_like(mean_b)

        feat = torch.cat([mean_b, std_b, mean_k, std_k, ones], dim=1)  # [B,5]
        h = F.gelu(self.fc1(feat))
        
        # 2. The Learned "Base" Gamma (for 32x32 scale)
        # Network learns material-dependent stiffness, roughly in (0.5, 2.0)
        base_gamma = 0.5 + 1.5 * torch.sigmoid(self.fc2(h))
        
        # 3. The Physics Enforcer (CFL Condition)
        # If resolution doubles (H: 32→64), step size must drop by 4x
        # This is universal physics, not something to learn
        cfl_scale = (32.0 / H) ** 2
        
        gamma = base_gamma * cfl_scale
        
        return gamma  # [B,1]


class CoreSolver(nn.Module):
    """
    Core DEQ update map f_theta with Physical State Preservation.
    
    THE KEY INSIGHT:
    Standard ResNets normalize the OUTPUT state z_new = norm(z + dz).
    This is catastrophic for PDEs because:
    - On large grids, CFL forces dz to be tiny
    - GroupNorm sees tiny updates as "variance drift" and erases them
    - The solver has SHORT-TERM MEMORY LOSS
    
    THE FIX: Pre-Norm Architecture
    - Normalize the FEATURES (hidden activations) for training stability
    - Leave the OUTPUT STATE (z) unnormalized for physics preservation
    - This transforms the model into a Neural Euler Integrator
    
    Result: Tiny physics updates accumulate naturally across hundreds of iterations.
    """
    def __init__(self, hidden_ch=32):
        super().__init__()
        in_ch = 1 + 3  # z, boundary, mask, k
        self.conv1 = nn.Conv2d(in_ch, hidden_ch, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(hidden_ch, 1, kernel_size=3, padding=1)
        
        # Norm is for HIDDEN features (the "thought vector"), not the physical state
        self.norm_h = nn.GroupNorm(4, hidden_ch)

    def forward_f(self, z, boundary, mask, k, alpha, gamma):
        """
        Pure Neural Euler Integration: z_{t+1} = z_t + γ α Δz
        
        z: [B,1,H,W] - Physical state (temperature, concentration, etc.)
        boundary, mask, k: [B,1,H,W] - Problem specification
        alpha: [B,1,H,W] - Per-pixel stabilizer (learned preconditioner)
        gamma: [B,1] - Global step size (CFL-scaled)
        """
        B, _, H, W = z.shape
        
        # 1. Concatenate physical state with boundary conditions
        # z is NEVER normalized - it's a physical quantity
        inp = torch.cat([z, boundary, mask, k], dim=1)
        
        # 2. Compute update via normalized CNN
        h = self.conv1(inp)
        h = self.norm_h(h)  # Normalize the "brain" (features), not the physics
        h = F.gelu(h)
        
        dz = self.conv2(h)  # [B,1,H,W] - Raw update proposal
        
        # 3. The Euler Step - Pure Physics Accumulation
        gamma_b = gamma.view(B, 1, 1, 1)
        
        # CRITICAL: No normalization after this step
        # Even microscopic updates (from CFL scaling) must persist
        z_new = z + gamma_b * alpha * dz
        
        # Enforce boundary conditions (physical constraint)
        z_new = z_new * (1.0 - mask) + boundary * mask
        return z_new


class MultiDEQPDEBlock(nn.Module):
    """
    Full 3-network DEQ block (heterogeneous PDE):
      z* = f_theta(z*, boundary, mask, k, alpha(boundary,mask,k), gamma(boundary,mask,k))
    """
    def __init__(self, hidden_ch=32, max_iter=50, tol=1e-4):
        super().__init__()
        self.core = CoreSolver(hidden_ch=hidden_ch)
        self.stabilizer = StabilizerNet(in_ch=3, hidden_ch=16)
        self.spec_ctrl = SpectralController(in_dim=5, hidden=32)  # Back to 5 (physics-informed scaling)
        self.max_iter = max_iter
        self.tol = tol

    @torch.no_grad()
    def solve_equilibrium(self, boundary, mask, k, z0=None):
        """
        Fixed-point iteration:
          z_{k+1} = f_theta(z_k, ...)
        using stabilizer + spectral controller at each step.
        """
        B, _, H, W = boundary.shape
        if z0 is None:
            z = boundary.clone()
        else:
            z = z0

        for _ in range(self.max_iter):
            alpha = self.stabilizer(boundary, mask, k)
            gamma = self.spec_ctrl(boundary, mask, k)
            z_next = self.core.forward_f(z, boundary, mask, k, alpha, gamma)
            diff = (z_next - z).reshape(B, -1).norm(dim=1).mean()
            z = z_next
            if diff < self.tol:
                break
        return z

    def forward(self, inp):
        """
        inp: [B,3,H,W] = [boundary, mask, k]
        For the differentiable forward pass, we:
          1. solve equilibrium in no-grad
          2. reattach graph by one application of f_theta with alpha,gamma
        """
        boundary, mask, k = inp[:, :1], inp[:, 1:2], inp[:, 2:3]
        with torch.no_grad():
            z_star = self.solve_equilibrium(boundary, mask, k)

        z_star = z_star.detach()
        z_star.requires_grad_(True)
        alpha = self.stabilizer(boundary, mask, k)
        gamma = self.spec_ctrl(boundary, mask, k)
        z_star = self.core.forward_f(z_star, boundary, mask, k, alpha, gamma)
        return z_star


# ============================================
# 3. Spectral proxy for this DEQ
# ============================================

def spectral_norm_proxy(core_solver, z_star, boundary, mask, k, alpha, gamma, n_power_iter=5):
    """
    Approximate largest singular value of J_f wrt z using power iteration.

    f(z) = core_solver.forward_f(z, boundary, mask, k, alpha, gamma)
    z_star: [B,1,H,W]
    alpha,gamma treated as constants (no grad for them).
    """
    B = z_star.size(0)

    # random initial v
    v = torch.randn_like(z_star)
    v_flat = v.reshape(B, -1)
    v_flat = v_flat / (v_flat.norm(dim=1, keepdim=True) + 1e-8)
    v = v_flat.view_as(z_star)

    for _ in range(n_power_iter):
        v.requires_grad_(True)
        y = core_solver.forward_f(z_star, boundary, mask, k, alpha, gamma)
        JTv, = torch.autograd.grad(
            outputs=y,
            inputs=z_star,
            grad_outputs=v,
            retain_graph=True,
            create_graph=True
        )
        JTv_flat = JTv.reshape(B, -1)
        v_flat = JTv_flat / (JTv_flat.norm(dim=1, keepdim=True) + 1e-8)
        v = v_flat.view_as(z_star)

    y = core_solver.forward_f(z_star, boundary, mask, k, alpha, gamma)
    JTv, = torch.autograd.grad(
        outputs=y,
        inputs=z_star,
        grad_outputs=v,
        retain_graph=True,
        create_graph=True
    )
    JTv_flat = JTv.reshape(B, -1)
    sigma = JTv_flat.norm(dim=1).mean()
    return sigma


def spectral_band_loss(phi,
                       lower=0.90,
                       upper=0.999,
                       alpha_low=0.5,
                       alpha_high=2.0,
                       beta=20.0):
    """
    Encourage phi to lie in [lower, upper] ⊂ (0,1).
    Wider band [0.90, 0.999] allows phi closer to critical boundary.
    """
    if phi < lower:
        return alpha_low * (lower - phi) ** 2
    elif lower <= phi <= upper:
        return phi.new_zeros(())
    elif upper < phi <= 1.0:
        return alpha_high * (phi - upper) ** 2
    else:
        return beta * (phi - 1.0) ** 2


# ============================================
# 4. Wrapper model + training loop
# ============================================

class MultiDEQPDEModel(nn.Module):
    def __init__(self, hidden_ch=32, max_iter=50, tol=1e-4):
        super().__init__()
        self.block = MultiDEQPDEBlock(hidden_ch=hidden_ch,
                                      max_iter=max_iter,
                                      tol=tol)

    def forward(self, inp):
        return self.block(inp)


def train_multi_deq_pde(
    H=32,
    W=32,
    n_train=1000,
    n_val=200,
    n_epochs=30,
    batch_size=32,
    lr=1e-3,
    lambda_spec=0.05,
):
    print(f"\n=== Training 3-network DEQ heterogeneous PDE model on {H}x{W} ===")

    train_ds = make_hetero_pde_dataset(n_train, H, W, n_iter=800, device=device, seed=0)
    val_ds = make_hetero_pde_dataset(n_val, H, W, n_iter=800, device=device, seed=1)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    model = MultiDEQPDEModel(hidden_ch=32, max_iter=50, tol=1e-4).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(1, n_epochs + 1):
        model.train()
        total_loss, total_phi, total_n, n_batches = 0.0, 0.0, 0, 0

        for xb, yb in train_loader:
            opt.zero_grad()
            pred = model(xb)
            task_loss = F.mse_loss(pred, yb)

            boundary, mask, k = xb[:, :1], xb[:, 1:2], xb[:, 2:3]
            with torch.no_grad():
                z_eq = model.block.solve_equilibrium(boundary, mask, k)
                alpha = model.block.stabilizer(boundary, mask, k)
                gamma = model.block.spec_ctrl(boundary, mask, k)

            z_eq = z_eq.detach()
            z_eq.requires_grad_(True)

            phi = spectral_norm_proxy(model.block.core, z_eq, boundary, mask, k, alpha, gamma)
            spec_loss = spectral_band_loss(phi)

            # warmup: skip spectral penalty for first 3 epochs
            if epoch <= 3:
                loss = task_loss
            else:
                loss = task_loss + lambda_spec * spec_loss

            loss.backward()
            opt.step()

            total_loss += loss.item() * xb.size(0)
            total_phi += float(phi.detach().cpu())
            total_n += xb.size(0)
            n_batches += 1

        train_loss = total_loss / total_n
        avg_phi = total_phi / max(1, n_batches)

        # validation
        model.eval()
        val_loss, val_n = 0.0, 0
        with torch.no_grad():
            for xb, yb in val_loader:
                pred = model(xb)
                loss = F.mse_loss(pred, yb)
                val_loss += loss.item() * xb.size(0)
                val_n += xb.size(0)
        val_loss /= val_n

        print(f"[3DEQ] Epoch {epoch:02d} | train_mse={train_loss:.6f} | "
              f"val_mse={val_loss:.6f} | phi≈{avg_phi:.3f}")

    return model


# ============================================
# 5. Simple resolution evaluation (using more iterations if needed)
# ============================================

@torch.no_grad()
def evaluate_multi_deq_on_grid(model, H, W, n_test=200, n_iter_gt=1000, base_iter=50):
    """
    Evaluate the DEQ PDE model on a grid HxW.
    
    CRITICAL: For larger grids with CFL-scaled gamma, we must:
    1. Scale iterations by (H/32)^2 to compensate for smaller steps
    2. Disable tolerance check (set tol=0) to prevent premature convergence
       (tiny CFL-scaled steps can trigger early stopping even when far from equilibrium)
    """
    print(f"\nEvaluating 3-network DEQ on {H}x{W} (heterogeneous PDE)...")
    test_ds = make_hetero_pde_dataset(n_test, H, W, n_iter=n_iter_gt, device=device, seed=123)
    test_loader = DataLoader(test_ds, batch_size=32)

    # 1. Scale iterations by (H/32)^2 because step size drops by (32/H)^2
    scale = (H / 32.0) ** 2
    new_iter = int(base_iter * scale)
    
    # Store old settings
    old_iter = model.block.max_iter
    old_tol = model.block.tol
    
    # 2. Apply evaluation settings
    model.block.max_iter = max(new_iter, base_iter)
    model.block.tol = 0.0  # CRITICAL: Disable early stopping for CFL-scaled steps
    
    total_loss, total_n = 0.0, 0
    for xb, yb in test_loader:
        boundary, mask, k = xb[:, :1], xb[:, 1:2], xb[:, 2:3]
        z_eq = model.block.solve_equilibrium(boundary, mask, k)
        loss = F.mse_loss(z_eq, yb)
        total_loss += loss.item() * xb.size(0)
        total_n += xb.size(0)

    # Restore original settings
    model.block.max_iter = old_iter
    model.block.tol = old_tol
    
    mse = total_loss / total_n
    print(f"[3DEQ] Test MSE on {H}x{W}: {mse:.6f} (ran {new_iter} iters, tol=0)")
    return mse


# ============================================
# 6. Main
# ============================================

def visualize_stabilizer(model, H=32, W=32, save_path="stabilizer_analysis.png"):
    """
    Visualize how the stabilizer network adapts to heterogeneous conductivity.
    
    The "money plot" that proves the cybernetic claim:
    - If alpha correlates with k (or its gradients), the network is reading material properties
    - If alpha is random noise, the theory is weak
    """
    import matplotlib.pyplot as plt
    
    # Generate one sample
    test_ds = make_hetero_pde_dataset(1, H, W, n_iter=800, device=device, seed=42)
    xb, yb = test_ds[0]
    xb = xb.unsqueeze(0).to(device)
    yb = yb.unsqueeze(0).to(device)
    
    boundary, mask, k = xb[:, :1], xb[:, 1:2], xb[:, 2:3]
    
    # Get the alpha map and solution
    with torch.no_grad():
        alpha = model.block.stabilizer(boundary, mask, k)
        z_eq = model.block.solve_equilibrium(boundary, mask, k)
        gamma = model.block.spec_ctrl(boundary, mask, k)
    
    # Move to CPU for plotting
    k_cpu = k[0, 0].cpu().numpy()
    z_eq_cpu = z_eq[0, 0].cpu().numpy()
    alpha_cpu = alpha[0, 0].cpu().numpy()
    boundary_cpu = boundary[0, 0].cpu().numpy()
    
    # Create figure
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. The Conductivity Map (The Problem)
    im0 = axs[0, 0].imshow(k_cpu, cmap='inferno', aspect='auto')
    axs[0, 0].set_title(f"Conductivity k(x,y)\n(Range: {k_cpu.min():.2f} - {k_cpu.max():.2f})", fontsize=12)
    axs[0, 0].set_xlabel("x")
    axs[0, 0].set_ylabel("y")
    plt.colorbar(im0, ax=axs[0, 0], fraction=0.046)
    
    # 2. The Solution (The Result)
    im1 = axs[0, 1].imshow(z_eq_cpu, cmap='viridis', aspect='auto')
    axs[0, 1].set_title(f"Equilibrium Solution u*\n(MSE from GT: {F.mse_loss(z_eq, yb).item():.6f})", fontsize=12)
    axs[0, 1].set_xlabel("x")
    axs[0, 1].set_ylabel("y")
    plt.colorbar(im1, ax=axs[0, 1], fraction=0.046)
    
    # 3. The Stabilizer Map (The Brain)
    im2 = axs[1, 0].imshow(alpha_cpu, cmap='plasma', vmin=0, vmax=1, aspect='auto')
    axs[1, 0].set_title(f"Learned Stabilizer α(x,y)\n(Mean: {alpha_cpu.mean():.3f}, Std: {alpha_cpu.std():.3f})", fontsize=12)
    axs[1, 0].set_xlabel("x")
    axs[1, 0].set_ylabel("y")
    plt.colorbar(im2, ax=axs[1, 0], fraction=0.046)
    
    # 4. Correlation Analysis (The Proof)
    # Mask out boundary for cleaner correlation
    mask_cpu = mask[0, 0].cpu().numpy()
    interior = mask_cpu == 0
    
    k_flat = k_cpu[interior].flatten()
    alpha_flat = alpha_cpu[interior].flatten()
    
    axs[1, 1].scatter(k_flat, alpha_flat, alpha=0.3, s=10, c='navy')
    axs[1, 1].set_xlabel("k(x,y) value", fontsize=11)
    axs[1, 1].set_ylabel("α(x,y) value", fontsize=11)
    axs[1, 1].set_title(f"Heterogeneous Adaptation\nγ={gamma.item():.3f}, Grid={H}×{W}", fontsize=12)
    axs[1, 1].grid(True, alpha=0.3)
    
    # Compute correlation
    import numpy as np
    corr = np.corrcoef(k_flat, alpha_flat)[0, 1]
    axs[1, 1].text(0.05, 0.95, f"Correlation: {corr:.3f}", 
                   transform=axs[1, 1].transAxes, fontsize=10,
                   verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n✅ Stabilizer visualization saved to: {save_path}")
    print(f"   Correlation(k, α) = {corr:.3f}")
    if abs(corr) > 0.3:
        print(f"   🎯 Strong adaptation detected! Network is reading material properties.")
    else:
        print(f"   ⚠️  Weak correlation - network may not be using k effectively.")
    
    return fig


if __name__ == "__main__":
    H_train = W_train = 32

    model = train_multi_deq_pde(
        H=H_train,
        W=W_train,
        n_train=1000,
        n_val=200,
        n_epochs=30,
        batch_size=32,
        lr=1e-3,
        lambda_spec=0.05,
    )

    print("\n=== Resolution generalization for 3-network DEQ (heterogeneous PDE) ===")
    for H_eval in [32, 64, 128]:
        W_eval = H_eval
        _ = evaluate_multi_deq_on_grid(model, H_eval, W_eval,
                                       n_test=200,
                                       n_iter_gt=1200,
                                       base_iter=50)
    
    # Visualize stabilizer adaptation
    print("\n=== Analyzing Heterogeneous Adaptation ===")
    visualize_stabilizer(model, H=32, W=32, save_path="stabilizer_32x32.png")
