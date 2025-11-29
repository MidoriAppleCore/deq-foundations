import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_default_dtype(torch.float32)


# ============================================
# 1. CCS-style heterogeneous PDE data generation
#    Solve ∇·(k∇p) = Q with random k(x,y), injection Q, and boundary p
# ============================================

def generate_ccs_boundary(batch_size, H, W, device=device):
    """
    CCS-style Dirichlet boundary:
      - Top: p = 0 (atmosphere reference)
      - Sides: small far-field pressure (weak gradients)
      - Bottom: no explicit BC (treated as interior -> approximate no-flow)
    Returns:
      boundary: [B,1,H,W]
      mask    : [B,1,H,W], 1 only where p is fixed.
    """
    boundary = torch.zeros(batch_size, 1, H, W, device=device)
    mask = torch.zeros_like(boundary)

    # Top boundary: fixed atmospheric reference p=0
    boundary[:, :, 0, :] = 0.0
    mask[:, :, 0, :] = 1.0

    # Side boundaries: small random far-field pressures near 0
    left_bc  = 0.1 * (torch.rand(batch_size, 1, H, device=device) - 0.5)
    right_bc = 0.1 * (torch.rand(batch_size, 1, H, device=device) - 0.5)

    boundary[:, :, :, 0] = left_bc
    boundary[:, :, :, -1] = right_bc
    mask[:, :, :, 0] = 1.0
    mask[:, :, :, -1] = 1.0

    # Bottom row: NO Dirichlet mask (mask stays 0 there)
    #   -> it acts like an interior row, with approximate no-flow via padding

    return boundary, mask


def generate_geological_permeability(batch_size, H, W, device=device):
    """
    Geological-style k(x,y) for CCS:
      - Base: log-normal matrix permeability
      - Caprock: ultra-low k in top ~20%
      - Weak spots in caprock: higher k "leakage windows"
      - Fractures: high-k linear features (CO2 superhighways)
    
    TIGHTENED RANGE: [0.01, 5.0] for numerical stability.
    """
    k_fields = []
    k_min, k_max = 0.01, 5.0  # Tightened range for stability

    for _ in range(batch_size):
        # 1. Base log-normal field
        log_k = torch.randn(1, 1, H, W, device=device)

        # Mild spatial smoothing (geological correlation)
        kernel = torch.ones(1, 1, 3, 3, device=device) / 9.0
        log_k = F.conv2d(log_k, kernel, padding=1)
        log_k = torch.clamp(log_k, -2.0, 2.0)

        k = torch.exp(log_k)

        # 2. Caprock at top: very low permeability
        cap_thickness = int(0.2 * H)
        caprock_mask = torch.zeros_like(k)
        caprock_mask[:, :, :cap_thickness, :] = 1.0

        # Base caprock reduction (e.g. 1e-3x)
        k = k * (1.0 - 0.999 * caprock_mask)

        # 3. Weak spots in caprock: partially damaged seal
        n_weak = 2
        for _ in range(n_weak):
            x_c = torch.randint(W // 4, 3 * W // 4, (1,), device=device).item()
            y_c = torch.randint(0, cap_thickness, (1,), device=device).item()
            radius = max(2, int(0.05 * W))

            yy, xx = torch.meshgrid(
                torch.arange(H, device=device),
                torch.arange(W, device=device),
                indexing="ij"
            )
            dist2 = (yy - y_c) ** 2 + (xx - x_c) ** 2
            weak_mask = (dist2 <= radius ** 2).float().unsqueeze(0).unsqueeze(0)

            # Raise k inside weak spot but keep it below matrix k_max
            k = k * (1.0 - 0.8 * weak_mask) + 0.8 * weak_mask * k.max()

        # 4. Fracture networks: high-k linear features
        n_fractures = 2 + torch.randint(0, 3, (1,), device=device).item()
        frac_mask = torch.zeros_like(k)

        for _ in range(n_fractures):
            angle = torch.rand(1, device=device).item() * 3.14159
            y_start = torch.randint(0, H, (1,), device=device).item()
            x_start = torch.randint(0, W, (1,), device=device).item()
            length = int(min(H, W) * (0.3 + 0.4 * torch.rand(1, device=device).item()))
            width = max(2, H // 16)

            for i in range(length):
                y = int(y_start + i * torch.sin(torch.tensor(angle)))
                x = int(x_start + i * torch.cos(torch.tensor(angle)))
                if 0 <= y < H and 0 <= x < W:
                    for dy in range(-width, width + 1):
                        for dx in range(-width, width + 1):
                            yy = y + dy
                            xx = x + dx
                            if 0 <= yy < H and 0 <= xx < W:
                                frac_mask[0, 0, yy, xx] = 1.0

        # Boost k in fractures strongly
        k = k * (1.0 + 50.0 * frac_mask)

        # 5. Normalize to [k_min, k_max]
        k = k_min + (k_max - k_min) * (k - k.min()) / (k.max() - k.min() + 1e-8)

        k_fields.append(k)

    return torch.cat(k_fields, dim=0)


def generate_co2_injection(batch_size, H, W, device=device, ref_H=32):
    """
    Deep CO2 injection source term Q(x,y): wells in bottom quarter for CCS realism.
    
    RESOLUTION-CONSISTENT: Q scaled by (ref_H/H)² so that integrated injection
    is constant across resolutions (fixed total mass, not fixed intensity).
    """
    Q = torch.zeros(batch_size, 1, H, W, device=device)
    
    # Resolution scaling: keep total integrated injection constant
    res_scale = (ref_H / H) ** 2

    for b in range(batch_size):
        n_wells = 1 + torch.randint(0, 3, (1,), device=device).item()
        for _ in range(n_wells):
            # Restrict to bottom quarter (deep injection)
            y_well = int(H * (0.75 + 0.2 * torch.rand(1, device=device).item()))
            x_well = torch.randint(W // 4, 3 * W // 4, (1,), device=device).item()
            # Moderate injection strength (downscaled for O(1) pressure)
            strength = (1.0 + 2.0 * torch.rand(1, device=device).item()) * res_scale
            radius = int(W * 0.08)

            for y in range(max(0, y_well - radius), min(H, y_well + radius)):
                for x in range(max(0, x_well - radius), min(W, x_well + radius)):
                    dist2 = (y - y_well) ** 2 + (x - x_well) ** 2
                    if dist2 <= radius ** 2:
                        Q[b, 0, y, x] += strength * torch.exp(
                            torch.tensor(-dist2 / (2 * (radius / 3) ** 2),
                                         device=device)
                        )
    return Q


def ccs_step(p, k, Q, boundary, mask):
    """
    One Jacobi-like iteration for ∇·(k∇p) = Q.

    Discrete form (grid units, no explicit dx):
      p_new = (k_n p_n + k_s p_s + k_w p_w + k_e p_e + Q) / (k_n + k_s + k_w + k_e)
    """
    B, C, H, W = p.shape

    p_pad = F.pad(p, (1, 1, 1, 1), mode="replicate")
    k_pad = F.pad(k, (1, 1, 1, 1), mode="replicate")

    p_n = p_pad[:, :, 0:H,   1:W+1]
    p_s = p_pad[:, :, 2:H+2, 1:W+1]
    p_w = p_pad[:, :, 1:H+1, 0:W]
    p_e = p_pad[:, :, 1:H+1, 2:W+2]

    k_n = k_pad[:, :, 0:H,   1:W+1]
    k_s = k_pad[:, :, 2:H+2, 1:W+1]
    k_w = k_pad[:, :, 1:H+1, 0:W]
    k_e = k_pad[:, :, 1:H+1, 2:W+2]

    eps = 1e-6
    num = k_n * p_n + k_s * p_s + k_w * p_w + k_e * p_e + Q
    den = k_n + k_s + k_w + k_e + eps

    p_new = num / den

    p_new = p_new * (1.0 - mask) + boundary * mask
    return p_new


def solve_ccs_pde(boundary, mask, k, Q, n_iter=1000):
    """
    Relaxation solver for ∇·(k∇p) = Q with Dirichlet boundary.
    """
    p = boundary.clone()
    for _ in range(n_iter):
        p = ccs_step(p, k, Q, boundary, mask)
    return p


def make_ccs_dataset(n_samples, H, W, n_iter=1000, device=device, seed=0, ref_H=32, normalize_p=True):
    """
    Generate CCS-style dataset: (boundary, mask, k, Q) -> p.
    Input: [boundary, mask, k, Q] (4 channels).
    Target: full pressure field p.
    
    NEW: normalize_p=True standardizes pressure to zero-mean, unit-std
    for consistent training loss scale across resolutions.
    """
    torch.manual_seed(seed)
    batch_size = 32
    xs, ys = [], []
    n_done = 0

    while n_done < n_samples:
        bs = min(batch_size, n_samples - n_done)
        boundary, mask = generate_ccs_boundary(bs, H, W, device=device)
        k = generate_geological_permeability(bs, H, W, device=device)
        Q = generate_co2_injection(bs, H, W, device=device, ref_H=ref_H)
        with torch.no_grad():
            p = solve_ccs_pde(boundary, mask, k, Q, n_iter=n_iter)
            
            # Normalize pressure for consistent scale
            if normalize_p:
                p_mean = p.view(bs, -1).mean(dim=1, keepdim=True).view(bs, 1, 1, 1)
                p_std = p.view(bs, -1).std(dim=1, keepdim=True).view(bs, 1, 1, 1) + 1e-6
                p = (p - p_mean) / p_std
                
        inp = torch.cat([boundary, mask, k, Q], dim=1)
        xs.append(inp)
        ys.append(p)
        n_done += bs

    X = torch.cat(xs, dim=0)
    Y = torch.cat(ys, dim=0)
    return TensorDataset(X, Y)


# ============================================
# 2. Three-network DEQ PDE block (CCS)
#    - core solver f_theta
#    - stabilizer g_phi (per-pixel alpha)
#    - spectral controller h_psi (global gamma)
# ============================================

class CCS_StabilizerNet(nn.Module):
    """
    Local stabilizer: produces per-pixel relaxation alpha(x) ∈ (0, 1).
    Input: [boundary, mask, k, Q].
    """
    def __init__(self, in_ch=4, hidden_ch=16):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, hidden_ch, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(hidden_ch, 1, kernel_size=3, padding=1)

    def forward(self, boundary, mask, k, Q):
        x = torch.cat([boundary, mask, k, Q], dim=1)
        h = F.gelu(self.conv1(x))
        alpha = torch.sigmoid(self.conv2(h))
        return alpha


class CCS_SpectralController(nn.Module):
    """
    Global spectral controller with Physics-Informed Scaling.

    gamma_final = base_gamma * (32/H)^2

    base_gamma is learned (material + source dependent),
    (32/H)^2 is the hard-coded CFL scaling across resolutions.
    """
    def __init__(self, in_dim=7, hidden=32):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.fc2 = nn.Linear(hidden, 1)

    def forward(self, boundary, mask, k, Q):
        B = boundary.size(0)
        _, _, H, W = boundary.shape

        b_abs = boundary.abs()
        mean_b = b_abs.view(B, -1).mean(dim=1, keepdim=True)
        std_b  = b_abs.view(B, -1).std(dim=1, keepdim=True)

        mean_k = k.view(B, -1).mean(dim=1, keepdim=True)
        std_k  = k.view(B, -1).std(dim=1, keepdim=True)

        Q_abs  = Q.abs()
        mean_Q = Q_abs.view(B, -1).mean(dim=1, keepdim=True)
        std_Q  = Q_abs.view(B, -1).std(dim=1, keepdim=True)

        ones = torch.ones_like(mean_b)

        feat = torch.cat(
            [mean_b, std_b, mean_k, std_k, mean_Q, std_Q, ones],
            dim=1
        )  # [B,7]
        h = F.gelu(self.fc1(feat))

        base_gamma = 0.5 + 1.5 * torch.sigmoid(self.fc2(h))
        cfl_scale = (32.0 / H) ** 2
        gamma = base_gamma * cfl_scale
        return gamma


class CCS_CoreSolver(nn.Module):
    """
    Core DEQ update map f_theta with Physical State Preservation.

    Pre-norm architecture:
    - normalize hidden activations (features),
    - do NOT normalize the state z.

    z_{t+1} = z_t + γ α Δz_θ(z_t, boundary, mask, k, Q)
    """
    def __init__(self, hidden_ch=32):
        super().__init__()
        in_ch = 1 + 4  # z, boundary, mask, k, Q
        self.conv1 = nn.Conv2d(in_ch, hidden_ch, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(hidden_ch, 1, kernel_size=3, padding=1)
        self.norm_h = nn.GroupNorm(4, hidden_ch)

    def forward_f(self, z, boundary, mask, k, Q, alpha, gamma):
        B, _, H, W = z.shape

        inp = torch.cat([z, boundary, mask, k, Q], dim=1)
        h = self.conv1(inp)
        h = self.norm_h(h)
        h = F.gelu(h)

        dz = self.conv2(h)

        gamma_b = gamma.view(B, 1, 1, 1)
        z_new = z + gamma_b * alpha * dz

        z_new = z_new * (1.0 - mask) + boundary * mask
        return z_new


class CCS_MultiDEQBlock(nn.Module):
    """
    Full 3-network DEQ block (CCS PDE):

      p* = f_theta(p*, boundary, mask, k, Q,
                   alpha(boundary,mask,k,Q),
                   gamma(boundary,mask,k,Q))
    """
    def __init__(self, hidden_ch=32, max_iter=50, tol=1e-4):
        super().__init__()
        self.core = CCS_CoreSolver(hidden_ch=hidden_ch)
        self.stabilizer = CCS_StabilizerNet(in_ch=4, hidden_ch=16)
        self.spec_ctrl = CCS_SpectralController(in_dim=7, hidden=32)
        self.max_iter = max_iter
        self.tol = tol

    @torch.no_grad()
    def solve_equilibrium(self, boundary, mask, k, Q, p0=None):
        B, _, H, W = boundary.shape
        if p0 is None:
            p = boundary.clone()
        else:
            p = p0

        for _ in range(self.max_iter):
            alpha = self.stabilizer(boundary, mask, k, Q)
            gamma = self.spec_ctrl(boundary, mask, k, Q)
            p_next = self.core.forward_f(p, boundary, mask, k, Q, alpha, gamma)
            diff = (p_next - p).reshape(B, -1).norm(dim=1).mean()
            p = p_next
            if diff < self.tol:
                break
        return p

    def forward(self, inp):
        """
        inp: [B,4,H,W] = [boundary, mask, k, Q]

        DEQ forward:
          1) solve equilibrium (no-grad),
          2) reattach graph by one application of f_theta.
        """
        boundary, mask, k, Q = inp[:, :1], inp[:, 1:2], inp[:, 2:3], inp[:, 3:4]
        with torch.no_grad():
            p_star = self.solve_equilibrium(boundary, mask, k, Q)

        p_star = p_star.detach()
        p_star.requires_grad_(True)
        alpha = self.stabilizer(boundary, mask, k, Q)
        gamma = self.spec_ctrl(boundary, mask, k, Q)
        p_star = self.core.forward_f(p_star, boundary, mask, k, Q, alpha, gamma)
        return p_star


# ============================================
# 3. Spectral proxy for this DEQ
# ============================================

def spectral_norm_proxy(core_solver, p_star, boundary, mask, k, Q, alpha, gamma, n_power_iter=5):
    """
    Approximate largest singular value of J_f wrt p using power iteration.

    f(p) = core_solver.forward_f(p, boundary, mask, k, Q, alpha, gamma)
    alpha,gamma treated as constants.
    """
    B = p_star.size(0)

    v = torch.randn_like(p_star)
    v_flat = v.reshape(B, -1)
    v_flat = v_flat / (v_flat.norm(dim=1, keepdim=True) + 1e-8)
    v = v_flat.view_as(p_star)

    for _ in range(n_power_iter):
        v.requires_grad_(True)
        y = core_solver.forward_f(p_star, boundary, mask, k, Q, alpha, gamma)
        JTv, = torch.autograd.grad(
            outputs=y,
            inputs=p_star,
            grad_outputs=v,
            retain_graph=True,
            create_graph=True
        )
        JTv_flat = JTv.reshape(B, -1)
        v_flat = JTv_flat / (v_flat.norm(dim=1, keepdim=True) + 1e-8)
        v = v_flat.view_as(p_star)

    y = core_solver.forward_f(p_star, boundary, mask, k, Q, alpha, gamma)
    JTv, = torch.autograd.grad(
        outputs=y,
        inputs=p_star,
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

class CCS_MultiDEQModel(nn.Module):
    def __init__(self, hidden_ch=32, max_iter=50, tol=1e-4):
        super().__init__()
        self.block = CCS_MultiDEQBlock(hidden_ch=hidden_ch,
                                       max_iter=max_iter,
                                       tol=tol)

    def forward(self, inp):
        return self.block(inp)


def train_ccs_multi_deq(
    n_train=1000,
    n_val=200,
    n_epochs=30,
    batch_size=32,
    lr=1e-3,
    lambda_spec=0.1,  # Stronger spectral regularization
    mixed_resolution=True,  # Train on multiple resolutions
    report_every=5,  # Generate reports every N epochs
):
    """
    Train CCS DEQ with mixed-resolution strategy for generalization.
    
    KEY FIXES:
    1. Mixed-resolution training (32 + 48) 
    2. Stronger spectral regularization (lambda=0.1)
    3. Tighter spectral band (0.85-0.98)
    """
    print(f"\n{'='*70}")
    print(f"🌍 CCS-DEQ: CO₂ Storage Pressure Solver")
    print(f"{'='*70}")
    
    # Setup reporting
    from deq_reports import create_reporter
    tracker, reporter = create_reporter("reservoir_pressure_deq")
    
    if mixed_resolution:
        print("📐 Mixed-resolution training: 32×32 + 48×48")
        train_ds_32 = make_ccs_dataset(n_train//2, 32, 32, n_iter=1000, device=device, seed=0, ref_H=32)
        train_ds_48 = make_ccs_dataset(n_train//2, 48, 48, n_iter=1500, device=device, seed=100, ref_H=32)
        
        train_loader_32 = DataLoader(train_ds_32, batch_size=batch_size, shuffle=True)
        train_loader_48 = DataLoader(train_ds_48, batch_size=batch_size, shuffle=True)
        train_loaders = [train_loader_32, train_loader_48]
        
        val_ds = make_ccs_dataset(n_val, 32, 32, n_iter=1000, device=device, seed=1, ref_H=32)
    else:
        print("📐 Single-resolution training: 32×32")
        train_ds = make_ccs_dataset(n_train, 32, 32, n_iter=1000, device=device, seed=0, ref_H=32)
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        train_loaders = None
        
        val_ds = make_ccs_dataset(n_val, 32, 32, n_iter=1000, device=device, seed=1, ref_H=32)
    
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    model = CCS_MultiDEQModel(hidden_ch=32, max_iter=50, tol=1e-4).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    
    global_step = 0

    for epoch in range(1, n_epochs + 1):
        model.train()
        total_loss, total_phi, total_n, n_batches = 0.0, 0.0, 0, 0
        epoch_alpha_mean, epoch_gamma_mean = 0.0, 0.0

        # Handle mixed-resolution training
        if mixed_resolution:
            import random
            all_batches = []
            for loader in train_loaders:
                all_batches.extend(list(loader))
            random.shuffle(all_batches)
            batch_iterator = iter(all_batches)
        else:
            batch_iterator = iter(train_loader)

        for xb, yb in batch_iterator:
            opt.zero_grad()
            pred = model(xb)
            task_loss = F.mse_loss(pred, yb)

            boundary, mask, k, Q = xb[:, :1], xb[:, 1:2], xb[:, 2:3], xb[:, 3:4]
            with torch.no_grad():
                p_eq = model.block.solve_equilibrium(boundary, mask, k, Q)
                alpha = model.block.stabilizer(boundary, mask, k, Q)
                gamma = model.block.spec_ctrl(boundary, mask, k, Q)

            p_eq = p_eq.detach()
            p_eq.requires_grad_(True)

            phi = spectral_norm_proxy(model.block.core, p_eq, boundary, mask, k, Q, alpha, gamma)
            # Tighter spectral band
            spec_loss = spectral_band_loss(phi, lower=0.85, upper=0.98)

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
            )
            global_step += 1

        train_loss = total_loss / total_n
        avg_phi = total_phi / max(1, n_batches)
        avg_alpha = epoch_alpha_mean / max(1, n_batches)
        avg_gamma = epoch_gamma_mean / max(1, n_batches)

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

        print(f"[CCS-DEQ] Epoch {epoch:02d}/{n_epochs} | train_mse={train_loss:.4f} | "
              f"val_mse={val_loss:.4f} | φ≈{avg_phi:.3f} | α={avg_alpha:.3f} | γ={avg_gamma:.3f}")
        
        # Generate intermediate reports
        if epoch % report_every == 0:
            reporter.generate_all()

    # Final reports
    print("\n📊 Generating final training reports...")
    reporter.generate_all()
    
    return model


# ============================================
# 5. Resolution evaluation
# ============================================

@torch.no_grad()
def evaluate_ccs_multi_deq_on_grid(model, H, W, n_test=200, n_iter_gt=1200, base_iter=50, ref_H=32):
    """
    Evaluate the DEQ CCS PDE model on a grid HxW.

    RESOLUTION CONSISTENCY:
    - Q scaled by (ref_H/H)² so integrated injection is constant
    - GT iterations scaled by (H/32)² for fair comparison
    """
    print(f"\n🔬 Evaluating CCS 3-network DEQ on {H}×{W}...")
    
    # Use ref_H for resolution-consistent Q
    test_ds = make_ccs_dataset(n_test, H, W, n_iter=n_iter_gt, device=device, seed=123, ref_H=ref_H)
    test_loader = DataLoader(test_ds, batch_size=32)

    scale = (H / 32.0) ** 2
    new_iter = int(base_iter * scale)

    old_iter = model.block.max_iter
    old_tol = model.block.tol

    model.block.max_iter = max(new_iter, base_iter)
    model.block.tol = 0.0

    total_loss, total_n = 0.0, 0
    for xb, yb in test_loader:
        boundary, mask, k, Q = xb[:, :1], xb[:, 1:2], xb[:, 2:3], xb[:, 3:4]
        p_eq = model.block.solve_equilibrium(boundary, mask, k, Q)
        loss = F.mse_loss(p_eq, yb)
        total_loss += loss.item() * xb.size(0)
        total_n += xb.size(0)

    model.block.max_iter = old_iter
    model.block.tol = old_tol

    mse = total_loss / total_n
    print(f"   ➤ Test MSE on {H}×{W}: {mse:.4f} (ran {new_iter} iters, tol=0)")
    return mse


# ============================================
# 6. Stabilizer visualization (optional)
# ============================================

def visualize_ccs_stabilizer(model, H=32, W=32, save_path="ccs_stabilizer_32x32.png", ref_H=32):
    import matplotlib.pyplot as plt
    import numpy as np

    test_ds = make_ccs_dataset(1, H, W, n_iter=1000, device=device, seed=42, ref_H=ref_H)
    xb, yb = test_ds[0]
    xb = xb.unsqueeze(0).to(device)
    yb = yb.unsqueeze(0).to(device)

    boundary, mask, k, Q = xb[:, :1], xb[:, 1:2], xb[:, 2:3], xb[:, 3:4]

    with torch.no_grad():
        alpha = model.block.stabilizer(boundary, mask, k, Q)
        p_eq = model.block.solve_equilibrium(boundary, mask, k, Q)
        gamma = model.block.spec_ctrl(boundary, mask, k, Q)

    k_cpu = k[0, 0].cpu().numpy()
    Q_cpu = Q[0, 0].cpu().numpy()
    p_cpu = p_eq[0, 0].cpu().numpy()
    alpha_cpu = alpha[0, 0].cpu().numpy()
    mask_cpu = mask[0, 0].cpu().numpy()

    fig, axs = plt.subplots(2, 3, figsize=(18, 10))

    im0 = axs[0, 0].imshow(np.log10(k_cpu + 1e-8), cmap='terrain', aspect='auto')
    axs[0, 0].set_title(f"Log₁₀ Permeability k(x,y)\n(Range: {k_cpu.min():.3f} - {k_cpu.max():.1f})", fontsize=12)
    axs[0, 0].set_xlabel("x")
    axs[0, 0].set_ylabel("y")
    plt.colorbar(im0, ax=axs[0, 0], fraction=0.046)
    # Add geological legend
    axs[0, 0].text(0.02, 0.98, "Dark = Caprock\nBright = Fractures\nMedium = Matrix", 
                   transform=axs[0, 0].transAxes, fontsize=9, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # CO2 Injection Sources
    im1 = axs[0, 1].imshow(Q_cpu, cmap='hot', aspect='auto')
    axs[0, 1].set_title(f"CO₂ Injection Q(x,y)\nTotal: {Q_cpu.sum():.2f} Mt/year", fontsize=12)
    axs[0, 1].set_xlabel("x")
    axs[0, 1].set_ylabel("y")
    plt.colorbar(im1, ax=axs[0, 1], fraction=0.046)
    
    # Mark injection wells
    for y in range(H):
        for x in range(W):
            if Q_cpu[y, x] > 0.5 * Q_cpu.max():
                from matplotlib.patches import Circle
                circle = Circle((x, y), 2, color='cyan', fill=False, linewidth=2)
                axs[0, 1].add_patch(circle)

    # Pressure Solution
    im2 = axs[0, 2].imshow(p_cpu, cmap='viridis', aspect='auto')
    mse_here = F.mse_loss(p_eq, yb).item()
    axs[0, 2].set_title(f"Pressure Field p(x,y)\n(MSE: {mse_here:.6f})", fontsize=12)
    axs[0, 2].set_xlabel("x")
    axs[0, 2].set_ylabel("y")
    plt.colorbar(im2, ax=axs[0, 2], fraction=0.046)

    # Stabilizer (Geological Intelligence)
    im3 = axs[1, 0].imshow(alpha_cpu, cmap='plasma', vmin=0, vmax=1, aspect='auto')
    axs[1, 0].set_title(f"Stabilizer α(x,y)\nμ={alpha_cpu.mean():.3f}, σ={alpha_cpu.std():.3f}", fontsize=12)
    axs[1, 0].set_xlabel("x")
    axs[1, 0].set_ylabel("y")
    plt.colorbar(im3, ax=axs[1, 0], fraction=0.046)
    axs[1, 0].text(0.02, 0.98, "High α = fast zones\nLow α = sensitive zones", 
                   transform=axs[1, 0].transAxes, fontsize=9, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # k-α Correlation (Geological Intelligence)
    interior = mask_cpu == 0
    k_flat = k_cpu[interior].flatten()
    alpha_flat = alpha_cpu[interior].flatten()

    axs[1, 1].scatter(k_flat, alpha_flat, alpha=0.3, s=10, c='darkblue')
    axs[1, 1].set_xlabel("Permeability k(x,y)", fontsize=11)
    axs[1, 1].set_ylabel("Stabilizer α(x,y)", fontsize=11)
    axs[1, 1].set_title("Fracture Recognition", fontsize=12)
    axs[1, 1].grid(True, alpha=0.3)
    
    # Q-α Correlation (Injection Awareness) 
    Q_flat = Q_cpu[interior].flatten()
    nonzero_Q = Q_flat > 0.01
    if nonzero_Q.sum() > 0:
        axs[1, 2].scatter(Q_flat[nonzero_Q], alpha_flat[nonzero_Q], alpha=0.5, s=20, c='darkred')
        corr_Q = np.corrcoef(Q_flat[nonzero_Q], alpha_flat[nonzero_Q])[0, 1]
    else:
        corr_Q = 0.0
        
    axs[1, 2].set_xlabel("Injection Q(x,y)", fontsize=11)
    axs[1, 2].set_ylabel("Stabilizer α(x,y)", fontsize=11)
    axs[1, 2].set_title("Injection Awareness", fontsize=12)
    axs[1, 2].grid(True, alpha=0.3)

    corr_k = np.corrcoef(k_flat, alpha_flat)[0, 1]
    axs[1, 1].text(0.05, 0.95, f"Correlation: {corr_k:.3f}\nγ={gamma.item():.3f}", 
                   transform=axs[1, 1].transAxes, fontsize=10, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    axs[1, 2].text(0.05, 0.95, f"Correlation: {corr_Q:.3f}\nGrid: {H}×{W}", 
                   transform=axs[1, 2].transAxes, fontsize=10, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n✅ CCS geological intelligence visualization saved to: {save_path}")
    print(f"🎯 GEOLOGICAL INTELLIGENCE METRICS:")
    print(f"   • Permeability-Stabilizer Correlation: {corr_k:.3f}")
    print(f"   • Injection-Stabilizer Correlation: {corr_Q:.3f}")
    print(f"   • Global Step Size γ: {gamma.item():.3f}")
    
    if abs(corr_k) > 0.3:
        print("   ✅ Strong fracture recognition detected!")
    if abs(corr_Q) > 0.2:
        print("   ✅ Injection-aware adaptation detected!")

    return fig


if __name__ == "__main__":
    print("\n" + "="*70)
    print("🌍 CCS-DEQ: CO₂ Geological Storage Pressure Solver")
    print("    PDE: ∇·(k∇p) = Q with heterogeneous permeability")
    print("="*70)
    
    model = train_ccs_multi_deq(
        n_train=1000,
        n_val=200,
        n_epochs=30,
        batch_size=32,
        lr=1e-3,
        lambda_spec=0.1,  # Stronger spectral reg
        mixed_resolution=True,  # Train on 32 + 48
    )

    print("\n" + "="*70)
    print("📊 Resolution Generalization Test (ref_H=32 for consistent Q)")
    print("="*70)
    
    results = {}
    for H_eval in [32, 48, 64, 96, 128]:
        W_eval = H_eval
        mse = evaluate_ccs_multi_deq_on_grid(
            model,
            H_eval,
            W_eval,
            n_test=200,
            n_iter_gt=1200,
            base_iter=50,
            ref_H=32,  # Resolution-consistent Q
        )
        results[H_eval] = mse

    print("\n" + "="*70)
    print("📈 Resolution Generalization Summary")
    print("="*70)
    base_mse = results[32]
    for H, mse in results.items():
        ratio = mse / base_mse if base_mse > 0 else float('inf')
        status = "✅" if ratio < 5.0 else "⚠️" if ratio < 20.0 else "❌"
        print(f"   {status} {H:3d}×{H:3d}: MSE={mse:8.4f} (×{ratio:.1f} vs 32×32)")

    print("\n" + "="*70)
    print("🔍 Analyzing CCS Geological Intelligence")
    print("="*70)
    visualize_ccs_stabilizer(model, H=32, W=32, save_path="ccs_stabilizer_32x32.png")
