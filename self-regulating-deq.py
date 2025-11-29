import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_default_dtype(torch.float32)


# ============================================
# 1. Laplace PDE data generation
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


def solve_laplace(boundary, mask, n_iter=500):
    """
    Jacobi relaxation to solve Δu = 0 with Dirichlet boundary.
    u: [B,1,H,W]
    """
    u = boundary.clone()

    kernel = torch.tensor(
        [[0.0, 1.0, 0.0],
         [1.0, 0.0, 1.0],
         [0.0, 1.0, 0.0]],
        device=u.device
    ).view(1, 1, 3, 3)

    for _ in range(n_iter):
        neigh_sum = F.conv2d(u, kernel, padding=1)
        u_new = neigh_sum / 4.0
        # enforce boundary
        u = u_new * (1.0 - mask) + boundary * mask

    return u


def make_pde_dataset(n_samples, H, W, n_iter=500, device=device, seed=0):
    """
    Generate (input, solution) pairs.
    Input: [boundary, mask] as 2 channels.
    Target: full solution u.
    """
    g = torch.Generator(device=device).manual_seed(seed)
    batch_size = 64
    xs, ys = [], []
    n_done = 0

    while n_done < n_samples:
        bs = min(batch_size, n_samples - n_done)
        boundary, mask = generate_random_boundary(bs, H, W, device=device)
        with torch.no_grad():
            u = solve_laplace(boundary, mask, n_iter=n_iter)
        inp = torch.cat([boundary, mask], dim=1)
        xs.append(inp)
        ys.append(u)
        n_done += bs

    X = torch.cat(xs, dim=0)
    Y = torch.cat(ys, dim=0)
    return TensorDataset(X, Y)


# ============================================
# 2. Three-network DEQ PDE block
#    - core solver f_theta
#    - stabilizer g_phi (per-pixel alpha)
#    - spectral controller h_psi (global gamma)
# ============================================

class StabilizerNet(nn.Module):
    """
    Local stabilizer: produces per-pixel relaxation alpha(x) ∈ (0, 1).
    Input: [boundary, mask] or [z, boundary, mask], but here keep it simple.
    """
    def __init__(self, in_ch=2, hidden_ch=16):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, hidden_ch, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(hidden_ch, 1, kernel_size=3, padding=1)

    def forward(self, boundary, mask):
        x = torch.cat([boundary, mask], dim=1)
        h = F.gelu(self.conv1(x))
        alpha = torch.sigmoid(self.conv2(h))  # (0,1)
        return alpha


class SpectralController(nn.Module):
    """
    Global spectral controller: maps a pooled feature vector to a scalar gamma>0
    that scales the update magnitude, shaping the Jacobian spectrum.
    """
    def __init__(self, in_dim=4, hidden=32):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.fc2 = nn.Linear(hidden, 1)

    def forward(self, boundary, mask):
        """
        Take simple global stats as features:
        - mean |boundary|
        - std |boundary|
        - mean mask (~boundary fraction)
        - constant 1
        """
        B = boundary.size(0)
        b_abs = boundary.abs()
        mean_b = b_abs.view(B, -1).mean(dim=1, keepdim=True)
        std_b = b_abs.view(B, -1).std(dim=1, keepdim=True)
        mean_m = mask.view(B, -1).mean(dim=1, keepdim=True)
        ones = torch.ones_like(mean_b)

        feat = torch.cat([mean_b, std_b, mean_m, ones], dim=1)  # [B,4]
        h = F.gelu(self.fc1(feat))
        # gamma > 0, around ~1
        gamma = 0.5 + torch.sigmoid(self.fc2(h))  # in (0.5,1.5)
        return gamma  # [B,1]


class CoreSolver(nn.Module):
    """
    Core DEQ update map f_theta: z_{k+1} = z_k + gamma * alpha * dz(z_k, boundary, mask)
    """
    def __init__(self, hidden_ch=32):
        super().__init__()
        in_ch = 1 + 2  # z, boundary, mask
        self.conv1 = nn.Conv2d(in_ch, hidden_ch, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(hidden_ch, 1, kernel_size=3, padding=1)
        self.norm = nn.GroupNorm(1, 1)

    def forward_f(self, z, boundary, mask, alpha, gamma):
        """
        z: [B,1,H,W]
        boundary, mask: [B,1,H,W]
        alpha: [B,1,H,W] per-pixel stabilizer
        gamma: [B,1] global scalar -> broadcast
        """
        B, _, H, W = z.shape
        inp = torch.cat([z, boundary, mask], dim=1)
        h = F.gelu(self.conv1(inp))
        dz = self.conv2(h)  # [B,1,H,W]

        gamma_b = gamma.view(B, 1, 1, 1)  # broadcast
        z_new = z + gamma_b * alpha * dz
        z_new = self.norm(z_new)
        # enforce boundary
        z_new = z_new * (1.0 - mask) + boundary * mask
        return z_new


class MultiDEQPDEBlock(nn.Module):
    """
    Full 3-network DEQ block:
      z* = f_theta(z*, boundary, mask, alpha(boundary,mask), gamma(boundary,mask))
    """
    def __init__(self, hidden_ch=32, max_iter=50, tol=1e-4):
        super().__init__()
        self.core = CoreSolver(hidden_ch=hidden_ch)
        self.stabilizer = StabilizerNet(in_ch=2, hidden_ch=16)
        self.spec_ctrl = SpectralController(in_dim=4, hidden=32)
        self.max_iter = max_iter
        self.tol = tol

    def forward_f_with_ctrl(self, z, boundary, mask):
        """
        Convenience: compute alpha, gamma (no grad if desired) and apply core.
        """
        alpha = self.stabilizer(boundary, mask)
        gamma = self.spec_ctrl(boundary, mask)
        z_new = self.core.forward_f(z, boundary, mask, alpha, gamma)
        return z_new, alpha, gamma

    @torch.no_grad()
    def solve_equilibrium(self, boundary, mask, z0=None):
        """
        Fixed-point iteration:
          z_{k+1} = f_theta(z_k, ...)
        using the stabilizer+spectral controller at each step.
        """
        B, _, H, W = boundary.shape
        if z0 is None:
            z = boundary.clone()
        else:
            z = z0

        for _ in range(self.max_iter):
            # Here we compute alpha,gamma in no-grad mode
            alpha = self.stabilizer(boundary, mask)
            gamma = self.spec_ctrl(boundary, mask)
            z_next = self.core.forward_f(z, boundary, mask, alpha, gamma)

            # simple convergence check
            diff = (z_next - z).reshape(B, -1).norm(dim=1).mean()
            z = z_next
            if diff < self.tol:
                break
        return z

    def forward(self, inp):
        """
        inp: [B,2,H,W] = [boundary, mask]
        For the differentiable forward pass, we:
          1. solve equilibrium in no-grad
          2. reattach graph by one application of f_theta with alpha,gamma
        """
        boundary, mask = inp[:, :1], inp[:, 1:2]
        with torch.no_grad():
            z_star = self.solve_equilibrium(boundary, mask)

        z_star = z_star.detach()
        z_star.requires_grad_(True)
        alpha = self.stabilizer(boundary, mask)
        gamma = self.spec_ctrl(boundary, mask)
        z_star = self.core.forward_f(z_star, boundary, mask, alpha, gamma)
        return z_star


# ============================================
# 3. Spectral proxy for this DEQ
# ============================================

def spectral_norm_proxy(core_solver, z_star, boundary, mask, alpha, gamma, n_power_iter=5):
    """
    Approximate largest singular value of J_f wrt z using power iteration.

    f(z) = core_solver.forward_f(z, boundary, mask, alpha, gamma)
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
        y = core_solver.forward_f(z_star, boundary, mask, alpha, gamma)
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

    y = core_solver.forward_f(z_star, boundary, mask, alpha, gamma)
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
    n_epochs=20,
    batch_size=32,
    lr=1e-3,
    lambda_spec=0.1,
    report_every=5,  # Generate reports every N epochs
):
    print(f"\n=== Training 3-network DEQ PDE model on {H}x{W} ===")
    
    # Setup reporting
    from deq_reports import create_reporter
    tracker, reporter = create_reporter(f"pde_deq_{H}x{W}")

    train_ds = make_pde_dataset(n_train, H, W, n_iter=500, device=device, seed=0)
    val_ds = make_pde_dataset(n_val, H, W, n_iter=500, device=device, seed=1)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    model = MultiDEQPDEModel(hidden_ch=32, max_iter=50, tol=1e-4).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    
    global_step = 0

    for epoch in range(1, n_epochs + 1):
        model.train()
        total_loss, total_phi, total_n, n_batches = 0.0, 0.0, 0, 0
        epoch_alpha_mean, epoch_gamma_mean = 0.0, 0.0

        for xb, yb in train_loader:
            opt.zero_grad()
            pred = model(xb)
            task_loss = F.mse_loss(pred, yb)

            # spectral regularization at equilibrium
            boundary, mask = xb[:, :1], xb[:, 1:2]
            with torch.no_grad():
                z_eq = model.block.solve_equilibrium(boundary, mask)
                # freeze alpha,gamma wrt z for the Jacobian approximation
                alpha = model.block.stabilizer(boundary, mask)
                gamma = model.block.spec_ctrl(boundary, mask)

            z_eq = z_eq.detach()
            z_eq.requires_grad_(True)

            phi = spectral_norm_proxy(model.block.core, z_eq, boundary, mask, alpha, gamma)
            spec_loss = spectral_band_loss(phi)

            # warmup: skip spectral penalty for first 3 epochs
            if epoch <= 3:
                loss = task_loss
                spec_loss_val = 0.0
            else:
                loss = task_loss + lambda_spec * spec_loss
                spec_loss_val = (lambda_spec * spec_loss).item() if hasattr(spec_loss, 'item') else float(spec_loss)

            loss.backward()
            opt.step()

            total_loss += loss.item() * xb.size(0)
            total_phi += float(phi.detach().cpu())
            total_n += xb.size(0)
            n_batches += 1
            
            # Track homeostatic vars
            epoch_alpha_mean += alpha.mean().item()
            epoch_gamma_mean += gamma.mean().item()
            
            # Record step metrics
            tracker.record(
                step=global_step,
                mse_loss=task_loss.item(),
                total_loss=loss.item(),
                phi=float(phi.detach().cpu()),
                spectral_penalty=spec_loss_val,
                alpha_mean=alpha.mean().item(),
                gamma_mean=gamma.mean().item(),
                alpha_std=alpha.std().item(),
                gamma_std=gamma.std().item() if gamma.numel() > 1 else 0.0,
            )
            global_step += 1

        train_loss = total_loss / total_n
        avg_phi = total_phi / max(1, n_batches)
        avg_alpha = epoch_alpha_mean / max(1, n_batches)
        avg_gamma = epoch_gamma_mean / max(1, n_batches)

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
        
        # Record validation
        tracker.record_val(global_step, loss=val_loss)

        print(f"[3DEQ] Epoch {epoch:02d} | train_mse={train_loss:.6f} | "
              f"val_mse={val_loss:.6f} | phi≈{avg_phi:.3f} | "
              f"α={avg_alpha:.3f} | γ={avg_gamma:.3f}")
        
        # Generate intermediate reports
        if epoch % report_every == 0:
            reporter.generate_all()

    # Final reports
    print("\n📊 Generating final training reports...")
    reporter.generate_all()
    
    return model


# ============================================
# 5. Simple resolution evaluation (using more iterations if needed)
# ============================================

@torch.no_grad()
def evaluate_multi_deq_on_grid(model, H, W, n_test=200, n_iter_gt=800, base_iter=50):
    """
    Evaluate the DEQ PDE model on a grid HxW.
    For harder (larger) grids, we give the equilibrium solver more iterations.
    """
    print(f"\nEvaluating 3-network DEQ on {H}x{W}...")
    test_ds = make_pde_dataset(n_test, H, W, n_iter=n_iter_gt, device=device, seed=123)
    test_loader = DataLoader(test_ds, batch_size=32)

    # choose max_iter scale heuristically: ~ (H/32)^2
    scale = (H / 32.0) ** 2
    new_iter = int(base_iter * scale)
    old_iter = model.block.max_iter
    model.block.max_iter = max(new_iter, base_iter)

    total_loss, total_n = 0.0, 0
    for xb, yb in test_loader:
        boundary, mask = xb[:, :1], xb[:, 1:2]
        z_eq = model.block.solve_equilibrium(boundary, mask)
        loss = F.mse_loss(z_eq, yb)
        total_loss += loss.item() * xb.size(0)
        total_n += xb.size(0)

    model.block.max_iter = old_iter
    mse = total_loss / total_n
    print(f"[3DEQ] Test MSE on {H}x{W}: {mse:.6f}")
    return mse


# ============================================
# 6. Main
# ============================================

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
        lambda_spec=0.05,  # reduced from 0.1 for gentler spectral shaping
    )

    print("\n=== Resolution generalization for 3-network DEQ ===")
    for H_eval in [32, 64, 128]:
        W_eval = H_eval
        _ = evaluate_multi_deq_on_grid(model, H_eval, W_eval,
                                       n_test=200,
                                       n_iter_gt=800,
                                       base_iter=50)
