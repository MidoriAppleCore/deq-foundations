import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
# REMOVED STATIC SCHEDULER IMPORT
# from torch.optim.lr_scheduler import StepLR 

from datasets import load_dataset
from transformers import AutoTokenizer

# ============================================================
# 1. DATA: WIKITEXT-2 + GPT-2 TOKENIZER
# ============================================================

class WikiTextSequenceDataset(Dataset):
    """
    Turn a long token stream into fixed-length language modeling sequences.
    Each item is (input_ids, target_ids) with a one-token shift.
    """
    def __init__(self, token_ids, seq_len):
        super().__init__()
        self.seq_len = seq_len
        # we need at least seq_len+1 tokens to form one example
        self.num_examples = (len(token_ids) - 1) // seq_len
        self.data = token_ids

    def __len__(self):
        return self.num_examples

    def __getitem__(self, idx):
        start = idx * self.seq_len
        end = start + self.seq_len + 1  # input + next-token target
        chunk = self.data[start:end]  # (seq_len+1,)
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        return x, y


def build_wikitext_dataloaders(
    seq_len=128,
    batch_size=16,
    min_train_tokens=None,
    min_valid_tokens=None,
    tokenizer_name="gpt2",
):
    """
    - Loads wikitext-2-raw-v1
    - Tokenizes with GPT-2 tokenizer
    - Flattens into one long sequence per split
    - Slices into fixed-length examples
    """
    print("Loading WikiText-2...")
    raw = load_dataset("wikitext", "wikitext-2-raw-v1")

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def tokenize_split(split):
        texts = "\n\n".join(raw[split]["text"])
        toks = tokenizer(
            texts,
            return_tensors=None,
            add_special_tokens=False
        )["input_ids"]
        return toks

    train_ids = tokenize_split("train")
    valid_ids = tokenize_split("validation")

    # Use full dataset
    if min_train_tokens is not None:
        train_ids = train_ids[:min_train_tokens]
    if min_valid_tokens is not None:
        valid_ids = valid_ids[:min_valid_tokens]


    print(f"Train tokens: {len(train_ids):,}, Valid tokens: {len(valid_ids):,}")

    train_ds = WikiTextSequenceDataset(train_ids, seq_len)
    valid_ds = WikiTextSequenceDataset(valid_ids, seq_len)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_ds, batch_size=batch_size, shuffle=False)

    vocab_size = tokenizer.vocab_size

    return train_loader, valid_loader, vocab_size, tokenizer


# ============================================================
# 2. DEQ TRANSFORMER BLOCK WITH Φ-REGULARIZATION
# ============================================================

class DEQTransformerBlock(nn.Module):
    """
    Single Transformer-style block:
      z_{t+1} = z_t + Attn(LN(z_t)) + MLP(LN(z_t)) + x_context
    """

    def __init__(self, dim, num_heads=4, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            batch_first=True,
            dropout=dropout
        )
        self.ln2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(dim * mlp_ratio), dim),
        )

    def forward(self, z, x_context, attn_mask=None):
        # z: (B, T, D), x_context: (B, T, D)

        # Self-attention with residual
        z_norm = self.ln1(z)
        attn_out, _ = self.attn(
            z_norm,
            z_norm,
            z_norm,
            attn_mask=attn_mask
        )
        z = z + attn_out

        # Inject input context (acts like a bias forcing dependence on x)
        z = z + x_context

        # MLP with residual
        z_norm = self.ln2(z)
        mlp_out = self.mlp(z_norm)
        z = z + mlp_out

        return z


def fixed_point_iteration(f, z0, max_iter=20, tol=1e-4, damping=0.5):
    """
    Simple damped Picard iteration:
        z_{k+1} = z_k + damping * (f(z_k) - z_k)
    
    This version uses fixed tolerance 'tol'.
    """
    z = z0
    with torch.no_grad():
        for _ in range(max_iter):
            f_z = f(z)
            diff = f_z - z
            err = diff.norm() / (z.norm() + 1e-6)
            z = z + damping * diff
            if err.item() < tol:
                break
    return z


def compute_spectral_radius(f, z, attn_mask=None, iters=4, eps=1e-3):
    """
    Estimate σ_max(J) where J = ∂f/∂z at given z via power iteration on J^T J.
    """
    z = z.detach().requires_grad_(True)
    f_base = f(z, attn_mask)  # (B, T, D)

    # Random unit vector of same shape as z
    v = torch.randn_like(z)
    v = v / (v.norm() + 1e-6)

    eigenvalue_sq = None

    for _ in range(iters):
        # Jv via finite difference
        z_perturbed = z + eps * v
        f_perturbed = f(z_perturbed, attn_mask)
        Jv = (f_perturbed - f_base) / eps

        # J^T J v via autograd: v_next = J^T (J v)
        grads = torch.autograd.grad(
            outputs=f_base,
            inputs=z,
            grad_outputs=Jv,
            retain_graph=True,
            allow_unused=False
        )[0]
        v_new = grads
        norm = v_new.norm()
        if norm.item() < 1e-8:
            break
        v = v_new / norm
        eigenvalue_sq = norm

    if eigenvalue_sq is None:
        eigenvalue_sq = torch.tensor(0.0, device=z.device)

    # Convert σ^2 → σ
    sigma = torch.sqrt(eigenvalue_sq + 1e-8)
    return sigma.detach()


def estimate_spectral_entropy(z_states: torch.Tensor) -> torch.Tensor:
    """
    Spectral entropy of covariance of z across (batch, time):
    """
    # z: (B, T, D)
    B, T, D = z_states.shape
    X = z_states.reshape(B * T, D)

    # Center
    X = X - X.mean(dim=0, keepdim=True)

    # Covariance ~ (D x D)
    N = X.shape[0]
    cov = X.T @ X / (max(N - 1, 1))

    # Symmetrize numerically
    cov = 0.5 * (cov + cov.T)

    # Eigenvalues
    eigvals = torch.linalg.eigvalsh(cov)
    eigvals = torch.clamp(eigvals, min=1e-8)

    p = eigvals / eigvals.sum()
    H = -(p * torch.log(p)).sum()
    return H  # scalar tensor on same device


class DEQLanguageModel(nn.Module):
    """
    DEQ language model:
    """

    def __init__(self, vocab_size, dim=512, num_heads=8, dropout=0.1, max_seq_len=512):
        super().__init__()
        self.dim = dim
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.token_emb = nn.Embedding(vocab_size, dim)
        self.pos_emb = nn.Embedding(max_seq_len, dim)
        self.deq_block = DEQTransformerBlock(dim, num_heads=num_heads, dropout=dropout)
        self.ln_out = nn.LayerNorm(dim)
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)
        self.attn_mask = None # Pre-computed mask for generation

    def _forward_deq(self, input_ids, compute_phi, phi_iters, tol):
        B, T = input_ids.shape
        device = input_ids.device

        # 1. Embeddings
        positions = torch.arange(T, device=device).unsqueeze(0).expand(B, T)
        x_emb = self.token_emb(input_ids) + self.pos_emb(positions)

        # 2. Causal Mask
        if self.attn_mask is None or self.attn_mask.size(0) != T:
            self.attn_mask = torch.triu(
                torch.ones(T, T, device=device, dtype=torch.bool), diagonal=1
            )
        causal_mask = self.attn_mask[:T, :T]

        # 3. Fixed Point Solver
        z0 = torch.zeros_like(x_emb)

        def f_no_mask(z):
            return self.deq_block(z, x_emb, attn_mask=causal_mask)

        # Use DYNAMIC tolerance controlled by the meta-learner
        z_star = fixed_point_iteration(
            lambda z: f_no_mask(z), z0, max_iter=20, tol=tol, damping=0.5
        )

        # 4. Final step (for gradient computation or output)
        if self.training:
            z_star = z_star.detach().requires_grad_(True)
            def f_with_mask(z, mask):
                return self.deq_block(z, x_emb, attn_mask=mask)
            z_out = f_with_mask(z_star, causal_mask)
        else:
            # When not training, z_out is the next state calculated from z_star
            z_out = self.deq_block(z_star, x_emb, attn_mask=causal_mask) 
            z_star = z_star.detach() # No need for grad tracking in eval

        # 5. Compute Φ (spectral radius)
        phi = torch.tensor(0.0, device=device)
        if compute_phi and self.training:
             # Need to re-define f_with_mask here for scope
             def f_with_mask(z, mask):
                 return self.deq_block(z, x_emb, attn_mask=mask)
             phi = compute_spectral_radius(
                f_with_mask, z_star, attn_mask=causal_mask, iters=phi_iters
            )
        
        # 6. LM head
        h = self.ln_out(z_out)
        logits = self.lm_head(h)
        # 7. Return z_out alongside z_star for residual calculation
        return logits, phi, z_star, z_out


    def forward(self, input_ids, compute_phi=True, phi_iters=4, tol=1e-4):
        """ Wrapper for training forward pass. """
        return self._forward_deq(input_ids, compute_phi, phi_iters, tol)

    @torch.no_grad()
    def generate(self, start_ids, max_new_tokens, temperature=0.7, tol=1e-4):
        """
        Auto-regressively generates new tokens using the DEQ forward pass.
        """
        self.eval()
        idx = start_ids.unsqueeze(0) # (1, T)
        
        for _ in range(max_new_tokens):
            
            # 1. Truncate sequence if it exceeds max_seq_len (critical for pos embeddings)
            idx_cond = idx if idx.size(1) <= self.max_seq_len else idx[:, -self.max_seq_len:]
            
            # 2. Forward pass (DEQ fixed-point solve)
            # Use dynamic tolerance for generation 
            logits, _, _, _ = self._forward_deq(idx_cond, compute_phi=False, phi_iters=0, tol=tol)
            
            # 3. Get logits for the last token and apply temperature
            logits = logits[:, -1, :] / temperature
            
            # 4. Sample next token
            probs = F.softmax(logits, dim=-1)
            next_token_id = torch.multinomial(probs, num_samples=1)
            
            # 5. Append to the sequence
            idx = torch.cat((idx, next_token_id), dim=1)
            
            # Stop if the sequence grows too long or reaches EOS (for completeness, though EOS is tokenizer-dependent)
            if idx.size(1) >= 2 * self.max_seq_len:
                 break
        
        self.train()
        return idx[0]


# ============================================================
# 3. GEOMETRY CONTROLLER (ψ) – meta-learner
# ============================================================

class GeometryController(nn.Module):
    """
    Geometry Controller: Now outputs 7 parameters (4 core + LR + TOL + gamma_res).
    Takes 5 inputs: phi_eff, H, lm_loss, val_loss_gap, overfit_signal
    """

    def __init__(self, hidden_dim=64):  # Increased hidden dim
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(5, hidden_dim),  # 5 inputs now
            nn.GELU(),  # GELU instead of Tanh for better gradients
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 7)  # 7 outputs
        )

    def forward(self, phi_eff, H, lm_loss, val_gap=None, overfit_ratio=None):
        """
        phi_eff, H, lm_loss are scalar tensors.
        val_gap: validation_loss - train_loss (overfitting signal)
        overfit_ratio: val_loss / train_loss (>1 means overfitting)
        Returns geometry parameters, LR, TOL, and gamma_res adjustment.
        """
        if val_gap is None:
            val_gap = torch.tensor(0.0, device=lm_loss.device)
        if overfit_ratio is None:
            overfit_ratio = torch.tensor(1.0, device=lm_loss.device)
            
        # Build feature vector (detach for stability)
        s = torch.stack([
            phi_eff.detach(), 
            H.detach(), 
            lm_loss.detach(),
            val_gap.detach(),
            overfit_ratio.detach()
        ], dim=0)  # (5,)
        s = s.unsqueeze(0)  # (1,5)
        out = self.mlp(s)   # (1,7)
        out = torch.tanh(out)[0]  # (7,), each in [-1,1]

        d_gamma_phi, d_gamma_man, d_phi_center, d_H_min, d_lr, d_tol, d_gamma_res = out

        return d_gamma_phi, d_gamma_man, d_phi_center, d_H_min, d_lr, d_tol, d_gamma_res


# ============================================================
# 4. TRAINING LOOP FOR WIKITEXT + Φ + MANIFOLD + META-CONTROLLER
# ============================================================

@dataclass
class TrainConfig:
    seq_len: int = 128
    batch_size: int = 16
    # --- INCREASED CAPACITY ---
    dim: int = 512
    num_heads: int = 8
    # -------------------------
    # --- DROPOUT ---
    dropout: float = 0.15 
    # ---------------------------------------
    lr: float = 4e-4 # BASE LR
    weight_decay: float = 0.01
    max_steps: int = 10000 
    log_every: int = 50
    eval_every: int = 200
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # Base Φ band in log-domain: phi_eff = log(1 + σ_max)
    gamma_phi_base: float = 0.02
    phi_low_base: float = 5.7
    phi_high_base: float = 6.0
    phi_iters: int = 3

    # Base manifold: one-sided lower wall on spectral entropy H(z*)
    gamma_manifold_base: float = 0.05
    H_min_base: float = 5.6 # Target entropy
    
    # Base output regularization weight (FIXED)
    gamma_output_base: float = 0.0005 

    # Base residual regularization weight (DYNAMICALLY CONTROLLED)
    gamma_residual_base: float = 0.1 

    # --- DYNAMIC TOLERANCE PARAMETERS ---
    tol_base: float = 1e-4
    tol_scale_factor: float = 0.5 
    # ------------------------------------

    # Meta-loss weights (for controller ψ) - INCREASED FOR STRONGER SIGNAL
    meta_vel_lambda: float = 10.0  # Was 1.0 - increase for stronger gradient signal
    meta_acc_lambda: float = 1.0   # Was 0.1 - increase for smoother learning
    controller_l2_reg: float = 1e-6  # Was 1e-5 - reduce regularization
    
    # Meta-controller learning rate (separate from model)
    meta_lr: float = 3e-3  # NEW: much higher LR for meta-network
    
    # LR Scaling Factor
    lr_scale_factor: float = 0.5
    
    # Residual Gamma Scaling Factor
    gamma_res_scale_factor: float = 0.5


def train_phi_deq_wikitext_with_meta():
    cfg = TrainConfig()
    cfg.max_steps = 10000 
    print("Config:", cfg)
    
    # Setup reporting
    from deq_reports import create_reporter
    tracker, reporter = create_reporter("llm_deq_meta")

    train_loader, valid_loader, vocab_size, tokenizer = build_wikitext_dataloaders(
        seq_len=cfg.seq_len,
        batch_size=cfg.batch_size,
        min_train_tokens=None,
        min_valid_tokens=None,
    )

    model = DEQLanguageModel(
        vocab_size=vocab_size,
        dim=cfg.dim,
        num_heads=cfg.num_heads,
        dropout=cfg.dropout,
        max_seq_len=cfg.seq_len,
    ).to(cfg.device)

    controller = GeometryController(hidden_dim=32).to(cfg.device)

    # Separate optimizers: θ (model) vs ψ (controller)
    model_opt = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay
    )

    controller_opt = torch.optim.Adam(
        controller.parameters(),
        lr=cfg.meta_lr  # Use the config's meta_lr (3e-3)
    )
    
    step = 0
    model.train()
    controller.train()

    # For meta-loss: store previous LM loss and velocity (as floats)
    prev_lm_loss = None
    prev_vel = 0.0
    
    # Track validation loss for meta-controller
    last_val_loss = None
    val_gap = torch.tensor(0.0, device=cfg.device)
    overfit_ratio = torch.tensor(1.0, device=cfg.device)
    
    # Sample prompt for periodic inference
    PROMPT = "The Deep Equilibrium Model is a new type of neural network that"
    PROMPT_IDS = torch.tensor(tokenizer.encode(PROMPT), dtype=torch.long).to(cfg.device)
    
    # Reduced temperature for less noisy generation
    INFERENCE_TEMP = 0.6 

    def evaluate():
        model.eval()
        total_loss = 0.0
        total_tokens = 0
        with torch.no_grad():
            # Use base tolerance for evaluation (consistent baseline)
            for xb, yb in valid_loader:
                xb = xb.to(cfg.device)
                yb = yb.to(cfg.device)
                # Pass fixed tolerance for validation
                logits, _, _, _ = model(xb, compute_phi=False, tol=cfg.tol_base)
                loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    yb.view(-1),
                    ignore_index=-100
                )
                total_loss += loss.item() * yb.numel()
                total_tokens += yb.numel()
        model.train()
        avg_loss = total_loss / total_tokens
        ppl = math.exp(avg_loss)
        return avg_loss, ppl

    print("Starting training...")
    while step < cfg.max_steps:
        for xb, yb in train_loader:
            xb = xb.to(cfg.device)
            yb = yb.to(cfg.device)
            
            # --- Get stats from previous step (Delayed Feedback) ---
            phi_eff_prev = torch.tensor(0.0, device=cfg.device) if prev_lm_loss is None else phi_eff
            H_prev = torch.tensor(0.0, device=cfg.device) if prev_lm_loss is None else H
            lm_loss_prev_detached = torch.tensor(0.0, device=cfg.device) if prev_lm_loss is None else lm_loss.detach()

            # ------------------------------------------------
            # Geometry controller ψ: outputs 7 core parameters (LR, TOL, gamma_res)
            # Now also receives val_gap and overfit_ratio for generalization awareness
            # ------------------------------------------------
            d_gamma_phi, d_gamma_man, d_phi_center, d_H_min, d_lr, d_tol, d_gamma_res = controller(
                phi_eff=phi_eff_prev, 
                H=H_prev, 
                lm_loss=lm_loss_prev_detached,
                val_gap=val_gap,
                overfit_ratio=overfit_ratio
            )

            # --- DYNAMIC PARAMETER CALCULATION ---
            phi_center_base = 0.5 * (cfg.phi_low_base + cfg.phi_high_base)
            phi_half_base = 0.5 * (cfg.phi_high_base - cfg.phi_low_base)

            gamma_phi = cfg.gamma_phi_base * torch.exp(0.5 * d_gamma_phi)
            gamma_manifold = cfg.gamma_manifold_base * torch.exp(0.5 * d_gamma_man)
            gamma_output = cfg.gamma_output_base  # FIXED
            
            # META-LEARNED RESIDUAL GAMMA (Differentiable Energy Investment)
            gamma_residual = cfg.gamma_residual_base * torch.exp(cfg.gamma_res_scale_factor * d_gamma_res)

            phi_center = phi_center_base + 0.2 * d_phi_center
            phi_half = phi_half_base
            H_min = cfg.H_min_base + 0.3 * d_H_min
            
            # META-LEARNED LEARNING RATE
            lr_scale = torch.exp(cfg.lr_scale_factor * d_lr)
            new_lr = cfg.lr * lr_scale
            
            # META-LEARNED TOLERANCE (RESOLUTION POLICY KNOB)
            # tol is applied in log space (exponential scale)
            tol_scale = torch.exp(cfg.tol_scale_factor * d_tol)
            dynamic_tol = cfg.tol_base * tol_scale

            # ----------------------------------

            # ------------------------------------------------
            # Forward pass: DEQ + φ + manifold stats (USING DYNAMIC TOLERANCE)
            # ------------------------------------------------
            # Pass DYNAMIC tolerance for training
            logits, phi_raw, z_star, z_out = model(xb, compute_phi=True, phi_iters=cfg.phi_iters, tol=dynamic_tol.item())

            # LM loss
            lm_loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                yb.view(-1),
                ignore_index=-100
            )

            # --- CALCULATE GEOMETRY METRICS FOR CURRENT STEP ---
            phi_eff = torch.log1p(phi_raw)
            H = estimate_spectral_entropy(z_star)
            residual_pen = (z_star - z_out).pow(2).mean() # Mean squared residual norm
            output_pen = z_star.pow(2).mean() # Mean squared energy (L2 norm^2)
            # --------------------------------------------------

            # ------------------------------------------------
            # Geometry penalties with *controlled* parameters
            # ------------------------------------------------
            # φ band penalty (Dynamic)
            phi_violation = torch.abs(phi_eff - phi_center) - phi_half
            phi_pen = F.relu(phi_violation) ** 2

            # Manifold lower wall (Dynamic)
            H_violation = H_min - H
            manifold_pen = F.relu(H_violation) ** 2
            
            # Output State Energy (FIXED)
            output_pen_term = gamma_output * output_pen
            
            # Residual Penalty (DYNAMICALLY CONTROLLED)
            residual_pen_term = gamma_residual * residual_pen

            total_inner_loss = (
                lm_loss
                + gamma_phi * phi_pen
                + gamma_manifold * manifold_pen
                + output_pen_term
                + residual_pen_term
            )

            # ------------------------------------------------
            # Meta-loss for ψ based on velocity / acceleration of LM loss
            # + overfitting penalty (val_gap)
            # ------------------------------------------------
            if prev_lm_loss is None:
                meta_loss = torch.zeros((), device=cfg.device)
                vel = 0.0
                acc = 0.0
            else:
                # velocity and acceleration wrt *task* loss history
                vel_t = lm_loss - prev_lm_loss
                acc_t = vel_t - prev_vel

                vel = vel_t.detach().item()
                acc = acc_t.detach().item()

                # Core meta-loss: penalize increasing loss and jerky dynamics
                meta_loss = (
                    cfg.meta_vel_lambda * F.relu(vel_t) ** 2
                    + cfg.meta_acc_lambda * (acc_t ** 2)
                )
                
                # Overfitting penalty: penalize large val-train gap
                # This encourages the controller to choose params that generalize
                overfit_penalty = 0.5 * F.relu(val_gap) ** 2
                meta_loss = meta_loss + overfit_penalty

            # Add a small regularization to controller outputs
            controller_output_reg = (
                d_gamma_phi**2 + d_gamma_man**2 + d_phi_center**2 + d_H_min**2 + d_lr**2 + d_tol**2 + d_gamma_res**2
            ).sum()
            meta_loss = meta_loss + cfg.controller_l2_reg * controller_output_reg

            # ------------------------------------------------
            # Backward: inner (θ) then meta (ψ)
            # ------------------------------------------------
            model_opt.zero_grad()
            controller_opt.zero_grad()

            total_inner_loss.backward(retain_graph=True)
            meta_loss.backward()

            # --- APPLY DYNAMIC LR BEFORE STEP ---
            for param_group in model_opt.param_groups:
                param_group['lr'] = new_lr.item()

            # Apply step
            model_opt.step()
            controller_opt.step()
            
            # ------------------------------------------------
            # Update history for meta-dynamics (detach to float)
            # ------------------------------------------------
            prev_lm_loss = lm_loss.detach().item()
            prev_vel = vel
            
            # ------------------------------------------------
            # Record metrics for reporting
            # ------------------------------------------------
            tracker.record(
                step=step,
                lm_loss=lm_loss.item(),
                total_loss=total_inner_loss.item(),
                phi_raw=phi_raw.item(),
                phi_eff=phi_eff.item(),
                spectral_entropy=H.item(),
                phi_penalty=(gamma_phi * phi_pen).item(),
                manifold_penalty=(gamma_manifold * manifold_pen).item(),
                residual_penalty=(gamma_residual * residual_pen).item(),
                output_penalty=output_pen_term.item(),
                gamma_phi=gamma_phi.item(),
                gamma_manifold=gamma_manifold.item(),
                gamma_residual=gamma_residual.item(),
                phi_center=phi_center.item(),
                H_min=H_min.item(),
                learning_rate=new_lr.item(),
                tolerance=dynamic_tol.item(),
                meta_loss=meta_loss.item(),
                velocity=vel,
                acceleration=acc,
            )

            step += 1

            if step % cfg.log_every == 0:
                with torch.no_grad():
                    # --- LOGGING ---
                    current_lr = model_opt.param_groups[0]['lr']
                    # Log gamma_res (the differentiable energy budget)
                    current_gamma_res = gamma_residual.item() 
                    
                    print(
                        f"Step {step:5d} | "
                        f"LM Loss: {lm_loss.item():.4f} | "
                        f"LR: {current_lr:.7f} | " # Log current LR
                        f"Tol: {dynamic_tol.item():.2e} | " # Log DYNAMIC tolerance
                        f"Φ_raw (σ_max): {phi_raw.item():.4f} | "
                        f"log(1+Φ): {phi_eff.item():.4f} | "
                        f"H(z*): {H.item():.4f} | "
                        f"M-pen: {(gamma_manifold * manifold_pen).item():.6f} | "
                        f"Res-pen: {(gamma_residual * residual_pen).item():.6f} | "
                        f"Gamma_Res: {current_gamma_res:.5f} | "
                        f"vel: {vel:.4f} | acc: {acc:.4f} | "
                        f"Inner Total: {total_inner_loss.item():.4f} | "
                        f"Meta Loss: {meta_loss.item():.6f}"
                    )
                    
                    # --- INFERENCE/GENERATION ---
                    generated_tokens = model.generate(
                        PROMPT_IDS, 
                        max_new_tokens=50, 
                        temperature=INFERENCE_TEMP,
                        tol=dynamic_tol.item() # Use dynamic tolerance for evaluation
                    )
                    generated_text = tokenizer.decode(generated_tokens.cpu().tolist())
                    print("-" * 80)
                    print(f"GENERATION @ Step {step}:")
                    print(generated_text)
                    print("-" * 80)


            if step % cfg.eval_every == 0:
                val_loss, val_ppl = evaluate()
                
                # Update overfitting signals for meta-controller
                last_val_loss = val_loss
                train_loss_avg = lm_loss.item()
                val_gap = torch.tensor(val_loss - train_loss_avg, device=cfg.device)
                overfit_ratio = torch.tensor(val_loss / max(train_loss_avg, 0.1), device=cfg.device)
                
                # Record validation metrics
                tracker.record_val(step, loss=val_loss, ppl=val_ppl, 
                                  val_gap=val_gap.item(), overfit_ratio=overfit_ratio.item())
                
                print(
                    f"  >> Eval @ step {step}: "
                    f"Val Loss={val_loss:.4f}, Val PPL={val_ppl:.2f}, "
                    f"ValGap={val_gap.item():.3f}, OverfitRatio={overfit_ratio.item():.3f}"
                )
                
                # Generate intermediate reports every 1000 steps
                if step % 1000 == 0:
                    reporter.generate_all()

            if step >= cfg.max_steps:
                break

    # Generate final reports
    print("\n📊 Generating final training reports...")
    reporter.generate_all()
    print("Training finished.")

if __name__ == "__main__":
    train_phi_deq_wikitext_with_meta()