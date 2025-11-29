"""
Universal DEQ Training Reports Module
=====================================
Comprehensive matplotlib reporting for all DEQ toy networks.
Tracks: spectral radius, entropy, penalties, meta-controller, homeostatic vars, etc.

Usage:
    from deq_reports import ReportManager, MetricsTracker
    
    tracker = MetricsTracker()
    reporter = ReportManager("my_experiment", tracker)
    
    # In training loop:
    tracker.record(step=step, lm_loss=loss.item(), phi=phi.item(), ...)
    
    # Periodically or at end:
    reporter.generate_all()
"""

import os
import math
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Callable
from datetime import datetime
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MaxNLocator
from matplotlib.colors import Normalize
import matplotlib.cm as cm

# Optional 3D support
try:
    from mpl_toolkits.mplot3d import Axes3D
    HAS_3D = True
except ImportError:
    HAS_3D = False


# ============================================================
# REPORT DIRECTORY MANAGEMENT
# ============================================================

def setup_report_dir(experiment_name: str, base_dir: str = "reports") -> str:
    """Create timestamped report directory for an experiment."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_dir = Path(base_dir) / f"{experiment_name}_{timestamp}"
    report_dir.mkdir(parents=True, exist_ok=True)
    print(f"📊 Reports will be saved to: {report_dir}/")
    return str(report_dir)


# ============================================================
# UNIVERSAL METRICS TRACKER
# ============================================================

@dataclass  
class MetricsTracker:
    """
    Universal metrics tracker for any DEQ training run.
    Dynamically accepts any metric via record().
    """
    
    # Core tracking dict - allows arbitrary metrics
    _data: Dict[str, List[float]] = field(default_factory=dict)
    
    # Special step tracking
    steps: List[int] = field(default_factory=list)
    val_steps: List[int] = field(default_factory=list)
    
    # Image/field snapshot storage (for PDE problems)
    _snapshots: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict)
    
    def record(self, step: int, **kwargs):
        """Record arbitrary metrics for a training step."""
        self.steps.append(step)
        for key, value in kwargs.items():
            if key not in self._data:
                self._data[key] = []
            # Handle tensors
            if hasattr(value, 'item'):
                value = value.item()
            elif hasattr(value, 'cpu'):
                value = value.cpu().numpy()
            self._data[key].append(float(value))
    
    def record_val(self, step: int, **kwargs):
        """Record validation metrics."""
        self.val_steps.append(step)
        for key, value in kwargs.items():
            val_key = f"val_{key}"
            if val_key not in self._data:
                self._data[val_key] = []
            if hasattr(value, 'item'):
                value = value.item()
            self._data[val_key].append(float(value))
    
    def get(self, key: str) -> List[float]:
        """Get a metric series."""
        return self._data.get(key, [])
    
    def has(self, key: str) -> bool:
        """Check if metric exists."""
        return key in self._data and len(self._data[key]) > 0
    
    def keys(self) -> List[str]:
        """List all tracked metrics."""
        return list(self._data.keys())
    
    def as_numpy(self) -> Dict[str, np.ndarray]:
        """Export all metrics as numpy arrays."""
        result = {'steps': np.array(self.steps), 'val_steps': np.array(self.val_steps)}
        for k, v in self._data.items():
            result[k] = np.array(v)
        return result
    
    def save(self, filepath: str):
        """Save metrics to npz file."""
        np.savez(filepath, **self.as_numpy())
        print(f"  ✓ Metrics saved to {filepath}")
    
    # --------------------------------------------------------
    # IMAGE/FIELD SNAPSHOT STORAGE (for PDE problems)
    # --------------------------------------------------------
    
    def store_snapshot(self, step: int, **fields):
        """
        Store 2D field snapshots for visualization.
        
        Example for reservoir pressure:
            tracker.store_snapshot(
                step=step,
                permeability=k[0,0].cpu().numpy(),
                injection=Q[0,0].cpu().numpy(),
                pressure=p[0,0].cpu().numpy(),
                alpha=alpha[0,0].cpu().numpy(),
                gamma=gamma.item(),
                prediction=pred[0,0].cpu().numpy(),
                target=target[0,0].cpu().numpy(),
            )
        """
        if 'field_snapshots' not in self._snapshots:
            self._snapshots['field_snapshots'] = []
        
        snapshot = {'step': step}
        for key, value in fields.items():
            # Handle PyTorch tensors
            if hasattr(value, 'cpu') and hasattr(value, 'numpy'):
                value = value.cpu().numpy()
            # Handle numpy arrays - keep as is
            elif isinstance(value, np.ndarray):
                pass  # Already numpy, keep it
            # Handle scalars (torch scalar or python number)
            elif hasattr(value, 'item'):
                try:
                    value = value.item()
                except (ValueError, RuntimeError):
                    # Multi-element tensor/array, convert to numpy
                    if hasattr(value, 'cpu'):
                        value = value.cpu().numpy()
                    elif hasattr(value, 'numpy'):
                        value = value.numpy()
            snapshot[key] = value
        
        self._snapshots['field_snapshots'].append(snapshot)
    
    def get_snapshots(self) -> List[Dict[str, Any]]:
        """Get all stored field snapshots."""
        return self._snapshots.get('field_snapshots', [])
    
    def has_snapshots(self) -> bool:
        """Check if any snapshots exist."""
        return len(self._snapshots.get('field_snapshots', [])) > 0


# ============================================================
# STYLE CONFIGURATION
# ============================================================

class PlotStyle:
    """Consistent styling for all plots."""
    
    colors = {
        'primary': '#2E86AB',
        'secondary': '#A23B72', 
        'tertiary': '#F18F01',
        'quaternary': '#C73E1D',
        'success': '#3A7D44',
        'warning': '#FFC107',
        'danger': '#DC3545',
        'info': '#17A2B8',
        'gray': '#6c757d',
    }
    
    # Color cycle for multiple series
    cycle = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#3A7D44', 
             '#17A2B8', '#6f42c1', '#fd7e14', '#20c997', '#e83e8c']
    
    @staticmethod
    def setup():
        """Apply global style settings."""
        try:
            plt.style.use('seaborn-v0_8-darkgrid')
        except:
            plt.style.use('ggplot')
    
    @staticmethod
    def smooth(data: List[float], window: int = 10) -> np.ndarray:
        """Apply moving average smoothing."""
        data = np.array(data)
        if len(data) < window:
            return data
        kernel = np.ones(window) / window
        return np.convolve(data, kernel, mode='valid')


# ============================================================
# REPORT GENERATOR
# ============================================================

class ReportManager:
    """
    Generates comprehensive training reports from MetricsTracker data.
    Automatically detects available metrics and generates appropriate plots.
    """
    
    def __init__(self, experiment_name: str, tracker: MetricsTracker, 
                 base_dir: str = "reports"):
        self.experiment_name = experiment_name
        self.tracker = tracker
        self.report_dir = setup_report_dir(experiment_name, base_dir)
        self.style = PlotStyle
        PlotStyle.setup()
    
    def generate_all(self):
        """Generate all available reports based on tracked metrics."""
        print(f"\n📈 Generating reports for '{self.experiment_name}'...")
        
        m = self.tracker
        generated = []
        
        # Always generate summary
        self._plot_summary_dashboard()
        generated.append("00_summary_dashboard.png")
        
        # Loss plots if we have loss data
        if m.has('loss') or m.has('lm_loss') or m.has('task_loss') or m.has('mse_loss'):
            self._plot_loss_curves()
            generated.append("01_loss_curves.png")
        
        # Spectral plots if we have phi/spectral data
        if m.has('phi') or m.has('phi_raw') or m.has('spectral_radius'):
            self._plot_spectral_dynamics()
            generated.append("02_spectral_dynamics.png")
        
        # Entropy/manifold if tracked
        if m.has('entropy') or m.has('spectral_entropy') or m.has('H'):
            self._plot_entropy_analysis()
            generated.append("03_entropy_analysis.png")
        
        # Penalty breakdown if we have penalty data
        penalty_keys = [k for k in m.keys() if 'penalty' in k.lower() or 'pen' in k.lower()]
        if penalty_keys:
            self._plot_penalty_breakdown(penalty_keys)
            generated.append("04_penalty_breakdown.png")
        
        # Meta-controller outputs
        gamma_keys = [k for k in m.keys() if k.startswith('gamma_') or k.startswith('d_')]
        if gamma_keys:
            self._plot_meta_controller(gamma_keys)
            generated.append("05_meta_controller.png")
        
        # Dynamic hyperparameters (LR, tolerance, etc.)
        hyperparam_keys = [k for k in m.keys() if k in ['learning_rate', 'lr', 'tolerance', 'tol']]
        if hyperparam_keys:
            self._plot_hyperparameters(hyperparam_keys)
            generated.append("06_hyperparameters.png")
        
        # Training dynamics (velocity, acceleration)
        if m.has('velocity') or m.has('vel'):
            self._plot_training_dynamics()
            generated.append("07_training_dynamics.png")
        
        # Homeostatic/stabilizer metrics
        homeostatic_keys = [k for k in m.keys() if 'alpha' in k.lower() or 'stabilizer' in k.lower()]
        if homeostatic_keys:
            self._plot_homeostatic(homeostatic_keys)
            generated.append("08_homeostatic.png")
        
        # Convergence metrics (iterations, residual)
        conv_keys = [k for k in m.keys() if 'iter' in k.lower() or 'residual' in k.lower() or 'converge' in k.lower()]
        if conv_keys:
            self._plot_convergence(conv_keys)
            generated.append("09_convergence.png")
        
        # Phase portrait if we have phi and entropy
        phi_key = next((k for k in ['phi_eff', 'phi', 'spectral_radius'] if m.has(k)), None)
        H_key = next((k for k in ['spectral_entropy', 'entropy', 'H'] if m.has(k)), None)
        loss_key = next((k for k in ['loss', 'lm_loss', 'task_loss', 'mse_loss'] if m.has(k)), None)
        if phi_key and H_key:
            self._plot_phase_portrait(phi_key, H_key, loss_key)
            generated.append("10_phase_portrait.png")
        
        # Validation analysis
        if m.val_steps:
            self._plot_validation()
            generated.append("11_validation.png")
        
        # PDE Field Snapshots (if stored)
        if m.has_snapshots():
            self._plot_pde_field_evolution()
            generated.append("12_pde_field_evolution.png")
            
            self._plot_pde_stabilizer_gallery()
            generated.append("13_pde_stabilizer_gallery.png")
            
            self._plot_pde_error_maps()
            generated.append("14_pde_error_maps.png")
            
            self._plot_pde_correlation_evolution()
            generated.append("15_pde_correlation_evolution.png")
        
        # Save raw metrics
        self.tracker.save(f"{self.report_dir}/metrics.npz")
        
        print(f"✅ Generated {len(generated)} reports in {self.report_dir}/")
        return generated
    
    # --------------------------------------------------------
    # INDIVIDUAL PLOT GENERATORS
    # --------------------------------------------------------
    
    def _plot_summary_dashboard(self):
        """Single-page summary of all key metrics."""
        m = self.tracker
        steps = np.array(m.steps)
        
        # Determine grid size based on available metrics
        n_plots = min(12, 4 + len([k for k in m.keys() if not k.startswith('val_')]))
        rows = 3
        cols = 4
        
        fig = plt.figure(figsize=(20, 12))
        gs = gridspec.GridSpec(rows, cols, figure=fig, hspace=0.35, wspace=0.3)
        fig.suptitle(f'{self.experiment_name} - Training Summary', fontsize=16, fontweight='bold')
        
        plot_idx = 0
        
        # Plot available core metrics
        core_metrics = [
            ('loss', 'lm_loss', 'task_loss', 'mse_loss'),  # Loss variants
            ('phi', 'phi_raw', 'spectral_radius'),  # Spectral
            ('entropy', 'spectral_entropy', 'H'),  # Entropy
            ('learning_rate', 'lr'),  # LR
            ('tolerance', 'tol'),  # Tolerance
            ('velocity', 'vel'),  # Dynamics
            ('gamma_phi',),  # Meta params
            ('gamma_residual', 'gamma_res'),
        ]
        
        for metric_group in core_metrics:
            if plot_idx >= rows * cols:
                break
            
            key = next((k for k in metric_group if m.has(k)), None)
            if key is None:
                continue
            
            ax = fig.add_subplot(gs[plot_idx // cols, plot_idx % cols])
            data = m.get(key)
            ax.plot(steps[:len(data)], data, alpha=0.6, color=self.style.cycle[plot_idx % len(self.style.cycle)])
            ax.set_title(key.replace('_', ' ').title(), fontsize=10)
            ax.tick_params(labelsize=8)
            
            # Log scale for certain metrics
            if 'loss' in key or 'lr' in key or 'tol' in key or 'gamma' in key:
                try:
                    ax.set_yscale('log')
                except:
                    pass
            
            plot_idx += 1
        
        # Add val loss if available
        if m.val_steps and m.has('val_loss'):
            ax = fig.add_subplot(gs[plot_idx // cols, plot_idx % cols])
            ax.plot(m.val_steps, m.get('val_loss'), 'o-', color=self.style.colors['secondary'])
            ax.set_title('Validation Loss', fontsize=10)
            ax.tick_params(labelsize=8)
        
        plt.savefig(f'{self.report_dir}/00_summary_dashboard.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("  ✓ 00_summary_dashboard.png")
    
    def _plot_loss_curves(self):
        """Detailed loss analysis."""
        m = self.tracker
        steps = np.array(m.steps)
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Loss Analysis', fontsize=16, fontweight='bold')
        
        # Find primary loss key
        loss_key = next((k for k in ['lm_loss', 'loss', 'task_loss', 'mse_loss'] if m.has(k)), None)
        
        if loss_key:
            loss = np.array(m.get(loss_key))
            
            # Raw + smoothed
            ax = axes[0, 0]
            ax.plot(steps, loss, alpha=0.3, color=self.style.colors['primary'], label='Raw')
            if len(loss) > 20:
                smoothed = self.style.smooth(loss, 20)
                ax.plot(steps[19:], smoothed, color=self.style.colors['primary'], linewidth=2, label='Smoothed')
            ax.set_xlabel('Step')
            ax.set_ylabel('Loss')
            ax.set_title(f'{loss_key} over Training')
            ax.legend()
            ax.set_yscale('log')
            
            # Train vs Val
            ax = axes[0, 1]
            ax.plot(steps, loss, alpha=0.5, color=self.style.colors['primary'], label='Train')
            if m.val_steps and m.has('val_loss'):
                ax.plot(m.val_steps, m.get('val_loss'), 'o-', color=self.style.colors['secondary'],
                       markersize=6, label='Val')
            ax.set_xlabel('Step')
            ax.set_ylabel('Loss')
            ax.set_title('Train vs Validation')
            ax.legend()
            
            # Loss histogram
            ax = axes[1, 0]
            ax.hist(loss, bins=50, color=self.style.colors['primary'], alpha=0.7, edgecolor='black')
            ax.axvline(np.median(loss), color='red', linestyle='--', label=f'Median: {np.median(loss):.4f}')
            ax.set_xlabel('Loss')
            ax.set_ylabel('Frequency')
            ax.set_title('Loss Distribution')
            ax.legend()
            
            # Loss rate of change
            ax = axes[1, 1]
            if len(loss) > 1:
                dloss = np.diff(loss)
                ax.plot(steps[1:], dloss, alpha=0.5, color=self.style.colors['tertiary'])
                ax.axhline(0, color='black', linestyle='-')
                ax.fill_between(steps[1:], 0, dloss, where=dloss > 0, alpha=0.3, color='red')
                ax.fill_between(steps[1:], 0, dloss, where=dloss < 0, alpha=0.3, color='green')
            ax.set_xlabel('Step')
            ax.set_ylabel('ΔLoss')
            ax.set_title('Loss Rate of Change')
        
        plt.tight_layout()
        plt.savefig(f'{self.report_dir}/01_loss_curves.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("  ✓ 01_loss_curves.png")
    
    def _plot_spectral_dynamics(self):
        """Spectral radius analysis."""
        m = self.tracker
        steps = np.array(m.steps)
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Spectral Dynamics (DEQ Stability)', fontsize=16, fontweight='bold')
        
        phi_key = next((k for k in ['phi_raw', 'phi', 'spectral_radius'] if m.has(k)), None)
        
        if phi_key:
            phi = np.array(m.get(phi_key))
            
            # Raw spectral radius
            ax = axes[0, 0]
            ax.plot(steps, phi, alpha=0.5, color=self.style.colors['quaternary'])
            if len(phi) > 20:
                smoothed = self.style.smooth(phi, 20)
                ax.plot(steps[19:], smoothed, color=self.style.colors['quaternary'], linewidth=2)
            ax.axhline(1.0, color='red', linestyle='--', linewidth=2, label='Critical: Φ=1')
            ax.fill_between(steps, 0.9, 1.0, alpha=0.2, color='green', label='Target Band')
            ax.set_xlabel('Step')
            ax.set_ylabel('Φ (Spectral Radius)')
            ax.set_title('Raw Spectral Radius')
            ax.legend()
            
            # log(1+phi) if available
            ax = axes[0, 1]
            if m.has('phi_eff'):
                phi_eff = np.array(m.get('phi_eff'))
                ax.plot(steps, phi_eff, alpha=0.5, color=self.style.colors['primary'])
                if len(phi_eff) > 20:
                    smoothed = self.style.smooth(phi_eff, 20)
                    ax.plot(steps[19:], smoothed, color=self.style.colors['primary'], linewidth=2)
                ax.set_ylabel('log(1 + Φ)')
            else:
                log_phi = np.log1p(phi)
                ax.plot(steps, log_phi, alpha=0.5, color=self.style.colors['primary'])
                ax.set_ylabel('log(1 + Φ)')
            ax.set_xlabel('Step')
            ax.set_title('Effective Spectral Radius (Log Domain)')
            
            # Phi distribution
            ax = axes[1, 0]
            ax.hist(phi, bins=50, color=self.style.colors['quaternary'], alpha=0.7, edgecolor='black')
            ax.axvline(1.0, color='red', linestyle='--', linewidth=2, label='Critical')
            ax.axvline(np.mean(phi), color='blue', linestyle='--', label=f'Mean: {np.mean(phi):.3f}')
            ax.set_xlabel('Φ')
            ax.set_ylabel('Frequency')
            ax.set_title('Spectral Radius Distribution')
            ax.legend()
            
            # Phi stability (rolling std)
            ax = axes[1, 1]
            if len(phi) > 50:
                window = 50
                rolling_std = [np.std(phi[max(0, i-window):i+1]) for i in range(len(phi))]
                ax.plot(steps, rolling_std, color=self.style.colors['info'])
                ax.set_ylabel('Rolling Std (window=50)')
            ax.set_xlabel('Step')
            ax.set_title('Spectral Stability')
        
        plt.tight_layout()
        plt.savefig(f'{self.report_dir}/02_spectral_dynamics.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("  ✓ 02_spectral_dynamics.png")
    
    def _plot_entropy_analysis(self):
        """Spectral entropy / manifold complexity analysis."""
        m = self.tracker
        steps = np.array(m.steps)
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle('Entropy / Manifold Complexity', fontsize=16, fontweight='bold')
        
        H_key = next((k for k in ['spectral_entropy', 'entropy', 'H'] if m.has(k)), None)
        
        if H_key:
            H = np.array(m.get(H_key))
            
            ax = axes[0]
            ax.plot(steps, H, alpha=0.5, color=self.style.colors['success'])
            if len(H) > 20:
                smoothed = self.style.smooth(H, 20)
                ax.plot(steps[19:], smoothed, color=self.style.colors['success'], linewidth=2)
            if m.has('H_min'):
                ax.plot(steps, m.get('H_min'), '--', color=self.style.colors['secondary'], 
                       alpha=0.7, label='H_min target')
            ax.set_xlabel('Step')
            ax.set_ylabel('H(z*)')
            ax.set_title('Spectral Entropy')
            ax.legend()
            
            ax = axes[1]
            ax.hist(H, bins=50, color=self.style.colors['success'], alpha=0.7, edgecolor='black')
            ax.axvline(np.mean(H), color='red', linestyle='--', label=f'Mean: {np.mean(H):.3f}')
            ax.set_xlabel('H(z*)')
            ax.set_ylabel('Frequency')
            ax.set_title('Entropy Distribution')
            ax.legend()
        
        plt.tight_layout()
        plt.savefig(f'{self.report_dir}/03_entropy_analysis.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("  ✓ 03_entropy_analysis.png")
    
    def _plot_penalty_breakdown(self, penalty_keys: List[str]):
        """Breakdown of all penalty terms."""
        m = self.tracker
        steps = np.array(m.steps)
        
        n_penalties = len(penalty_keys)
        cols = min(3, n_penalties)
        rows = (n_penalties + cols - 1) // cols + 1  # +1 for stacked plot
        
        fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))
        fig.suptitle('Penalty Term Breakdown', fontsize=16, fontweight='bold')
        axes = np.atleast_2d(axes)
        
        # Individual penalties
        for i, key in enumerate(penalty_keys):
            row, col = i // cols, i % cols
            ax = axes[row, col]
            data = np.array(m.get(key))
            ax.plot(steps[:len(data)], data, color=self.style.cycle[i % len(self.style.cycle)], alpha=0.7)
            ax.fill_between(steps[:len(data)], 0, data, alpha=0.3, color=self.style.cycle[i % len(self.style.cycle)])
            ax.set_xlabel('Step')
            ax.set_ylabel(key)
            ax.set_title(key.replace('_', ' ').title())
            ax.set_yscale('symlog', linthresh=1e-6)
        
        # Hide unused axes
        for i in range(n_penalties, rows * cols - cols):
            row, col = i // cols, i % cols
            axes[row, col].set_visible(False)
        
        # Stacked area chart in bottom row
        ax = fig.add_subplot(rows, 1, rows)
        penalty_data = []
        valid_keys = []
        for key in penalty_keys:
            data = m.get(key)
            if len(data) == len(steps):
                penalty_data.append(np.abs(data))
                valid_keys.append(key)
        
        if penalty_data:
            ax.stackplot(steps, penalty_data, labels=valid_keys,
                        colors=self.style.cycle[:len(valid_keys)], alpha=0.7)
            ax.set_xlabel('Step')
            ax.set_ylabel('Total Penalty')
            ax.set_title('Stacked Penalty Contributions')
            ax.legend(loc='upper right', fontsize=8)
        
        plt.tight_layout()
        plt.savefig(f'{self.report_dir}/04_penalty_breakdown.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("  ✓ 04_penalty_breakdown.png")
    
    def _plot_meta_controller(self, gamma_keys: List[str]):
        """Meta-controller / geometry controller outputs."""
        m = self.tracker
        steps = np.array(m.steps)
        
        n_params = len(gamma_keys)
        cols = min(3, n_params)
        rows = (n_params + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))
        fig.suptitle('Meta-Controller Outputs (ψ)', fontsize=16, fontweight='bold')
        
        if rows == 1 and cols == 1:
            axes = np.array([[axes]])
        elif rows == 1:
            axes = axes.reshape(1, -1)
        elif cols == 1:
            axes = axes.reshape(-1, 1)
        
        for i, key in enumerate(gamma_keys):
            row, col = i // cols, i % cols
            ax = axes[row, col]
            data = np.array(m.get(key))
            ax.plot(steps[:len(data)], data, color=self.style.cycle[i % len(self.style.cycle)])
            ax.set_xlabel('Step')
            ax.set_ylabel(key)
            ax.set_title(key.replace('_', ' ').title())
            if 'gamma' in key.lower():
                try:
                    ax.set_yscale('log')
                except:
                    pass
        
        # Hide unused
        for i in range(n_params, rows * cols):
            row, col = i // cols, i % cols
            axes[row, col].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(f'{self.report_dir}/05_meta_controller.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("  ✓ 05_meta_controller.png")
    
    def _plot_hyperparameters(self, keys: List[str]):
        """Dynamic hyperparameters (LR, tolerance)."""
        m = self.tracker
        steps = np.array(m.steps)
        
        fig, axes = plt.subplots(1, len(keys), figsize=(6*len(keys), 5))
        fig.suptitle('Dynamic Hyperparameters', fontsize=16, fontweight='bold')
        
        if len(keys) == 1:
            axes = [axes]
        
        for i, key in enumerate(keys):
            ax = axes[i]
            data = np.array(m.get(key))
            ax.plot(steps[:len(data)], data, color=self.style.colors['primary'], linewidth=2)
            ax.set_xlabel('Step')
            ax.set_ylabel(key)
            ax.set_title(key.replace('_', ' ').title())
            ax.set_yscale('log')
        
        plt.tight_layout()
        plt.savefig(f'{self.report_dir}/06_hyperparameters.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("  ✓ 06_hyperparameters.png")
    
    def _plot_training_dynamics(self):
        """Velocity and acceleration of loss."""
        m = self.tracker
        steps = np.array(m.steps)
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Training Dynamics', fontsize=16, fontweight='bold')
        
        vel_key = 'velocity' if m.has('velocity') else 'vel'
        vel = np.array(m.get(vel_key))
        
        # Velocity
        ax = axes[0, 0]
        ax.plot(steps[:len(vel)], vel, alpha=0.5, color=self.style.colors['primary'])
        ax.axhline(0, color='black', linestyle='-')
        ax.fill_between(steps[:len(vel)], 0, vel, where=np.array(vel) > 0, alpha=0.3, color='red')
        ax.fill_between(steps[:len(vel)], 0, vel, where=np.array(vel) < 0, alpha=0.3, color='green')
        ax.set_xlabel('Step')
        ax.set_ylabel('Velocity (ΔL)')
        ax.set_title('Loss Velocity')
        
        # Acceleration
        acc_key = 'acceleration' if m.has('acceleration') else 'acc'
        if m.has(acc_key):
            acc = np.array(m.get(acc_key))
            ax = axes[0, 1]
            ax.plot(steps[:len(acc)], acc, alpha=0.5, color=self.style.colors['secondary'])
            ax.axhline(0, color='black', linestyle='-')
            ax.set_xlabel('Step')
            ax.set_ylabel('Acceleration (Δ²L)')
            ax.set_title('Loss Acceleration')
        
        # Velocity histogram
        ax = axes[1, 0]
        ax.hist(vel, bins=50, color=self.style.colors['primary'], alpha=0.7, edgecolor='black')
        ax.axvline(0, color='red', linestyle='--')
        ax.set_xlabel('Velocity')
        ax.set_ylabel('Frequency')
        ax.set_title('Velocity Distribution')
        
        # Phase portrait
        if m.has(acc_key):
            ax = axes[1, 1]
            acc = np.array(m.get(acc_key))
            min_len = min(len(vel), len(acc))
            colors = np.linspace(0, 1, min_len)
            scatter = ax.scatter(vel[:min_len], acc[:min_len], c=colors, cmap='plasma', alpha=0.5, s=10)
            ax.axhline(0, color='gray', linestyle='-', alpha=0.5)
            ax.axvline(0, color='gray', linestyle='-', alpha=0.5)
            ax.set_xlabel('Velocity')
            ax.set_ylabel('Acceleration')
            ax.set_title('Dynamics Phase Portrait')
            plt.colorbar(scatter, ax=ax, label='Progress')
        
        plt.tight_layout()
        plt.savefig(f'{self.report_dir}/07_training_dynamics.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("  ✓ 07_training_dynamics.png")
    
    def _plot_homeostatic(self, keys: List[str]):
        """Homeostatic/stabilizer parameters (alpha, etc.)."""
        m = self.tracker
        steps = np.array(m.steps)
        
        n_params = len(keys)
        cols = min(3, n_params)
        rows = (n_params + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))
        fig.suptitle('Homeostatic / Stabilizer Metrics', fontsize=16, fontweight='bold')
        
        if n_params == 1:
            axes = np.array([[axes]])
        elif rows == 1:
            axes = axes.reshape(1, -1)
        
        for i, key in enumerate(keys):
            row, col = i // cols, i % cols
            ax = axes[row, col] if rows > 1 else axes[0, col]
            data = np.array(m.get(key))
            ax.plot(steps[:len(data)], data, color=self.style.cycle[i % len(self.style.cycle)])
            ax.set_xlabel('Step')
            ax.set_ylabel(key)
            ax.set_title(key.replace('_', ' ').title())
        
        plt.tight_layout()
        plt.savefig(f'{self.report_dir}/08_homeostatic.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("  ✓ 08_homeostatic.png")
    
    def _plot_convergence(self, keys: List[str]):
        """Convergence metrics (iterations, residual norm)."""
        m = self.tracker
        steps = np.array(m.steps)
        
        fig, axes = plt.subplots(1, len(keys), figsize=(6*len(keys), 5))
        fig.suptitle('Convergence Metrics', fontsize=16, fontweight='bold')
        
        if len(keys) == 1:
            axes = [axes]
        
        for i, key in enumerate(keys):
            ax = axes[i]
            data = np.array(m.get(key))
            ax.plot(steps[:len(data)], data, color=self.style.cycle[i % len(self.style.cycle)])
            ax.set_xlabel('Step')
            ax.set_ylabel(key)
            ax.set_title(key.replace('_', ' ').title())
            if 'residual' in key.lower():
                ax.set_yscale('log')
        
        plt.tight_layout()
        plt.savefig(f'{self.report_dir}/09_convergence.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("  ✓ 09_convergence.png")
    
    def _plot_phase_portrait(self, phi_key: str, H_key: str, loss_key: Optional[str]):
        """2D/3D phase portrait in (Φ, H, Loss) space."""
        m = self.tracker
        
        phi = np.array(m.get(phi_key))
        H = np.array(m.get(H_key))
        min_len = min(len(phi), len(H))
        phi, H = phi[:min_len], H[:min_len]
        
        fig = plt.figure(figsize=(14, 6))
        
        # 3D if available and we have loss
        if HAS_3D and loss_key and m.has(loss_key):
            loss = np.array(m.get(loss_key))[:min_len]
            
            ax1 = fig.add_subplot(121, projection='3d')
            colors = np.linspace(0, 1, min_len)
            scatter = ax1.scatter(phi, H, loss, c=colors, cmap='viridis', s=5, alpha=0.6)
            ax1.set_xlabel(phi_key)
            ax1.set_ylabel(H_key)
            ax1.set_zlabel(loss_key)
            ax1.set_title('3D Training Trajectory')
            
            ax2 = fig.add_subplot(122)
            scatter = ax2.scatter(phi, H, c=loss, cmap='hot_r', s=20, alpha=0.6)
            ax2.set_xlabel(phi_key)
            ax2.set_ylabel(H_key)
            ax2.set_title('Phase Space (colored by loss)')
            plt.colorbar(scatter, ax=ax2, label=loss_key)
        else:
            ax = fig.add_subplot(111)
            colors = np.linspace(0, 1, min_len)
            scatter = ax.scatter(phi, H, c=colors, cmap='viridis', s=20, alpha=0.6)
            ax.set_xlabel(phi_key)
            ax.set_ylabel(H_key)
            ax.set_title('Spectral Phase Space')
            plt.colorbar(scatter, ax=ax, label='Training Progress')
        
        plt.tight_layout()
        plt.savefig(f'{self.report_dir}/10_phase_portrait.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("  ✓ 10_phase_portrait.png")
    
    def _plot_validation(self):
        """Validation metrics over time."""
        m = self.tracker
        
        val_keys = [k for k in m.keys() if k.startswith('val_')]
        
        fig, axes = plt.subplots(1, len(val_keys), figsize=(6*len(val_keys), 5))
        fig.suptitle('Validation Metrics', fontsize=16, fontweight='bold')
        
        if len(val_keys) == 1:
            axes = [axes]
        
        for i, key in enumerate(val_keys):
            ax = axes[i]
            data = m.get(key)
            ax.plot(m.val_steps[:len(data)], data, 'o-', color=self.style.cycle[i % len(self.style.cycle)],
                   markersize=6)
            ax.set_xlabel('Step')
            ax.set_ylabel(key.replace('val_', ''))
            ax.set_title(key.replace('val_', 'Val ').replace('_', ' ').title())
        
        plt.tight_layout()
        plt.savefig(f'{self.report_dir}/11_validation.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("  ✓ 11_validation.png")

    # --------------------------------------------------------
    # PDE FIELD VISUALIZATION METHODS
    # --------------------------------------------------------
    
    def _plot_pde_field_evolution(self):
        """Show evolution of PDE fields over training (input, prediction, target)."""
        snapshots = self.tracker.get_snapshots()
        if not snapshots:
            return
        
        # Select evenly-spaced snapshots to show evolution
        n_show = min(5, len(snapshots))
        indices = np.linspace(0, len(snapshots)-1, n_show, dtype=int)
        selected = [snapshots[i] for i in indices]
        
        # Detect what fields are available
        sample = selected[0]
        field_types = []
        
        # Input fields
        if 'permeability' in sample:
            field_types.append(('permeability', None, 'terrain', 'Log₁₀ Permeability k'))
        if 'injection' in sample:
            field_types.append(('injection', None, 'hot', 'CO₂ Injection Q'))
        if 'boundary' in sample:
            field_types.append(('boundary', None, 'coolwarm', 'Boundary Conditions'))
        if 'mask' in sample:
            field_types.append(('mask', None, 'gray', 'Domain Mask'))
        
        # Output fields
        if 'prediction' in sample:
            field_types.append(('prediction', 'pressure', 'viridis', 'Prediction'))
        elif 'pressure' in sample:
            field_types.append(('pressure', None, 'viridis', 'Pressure p(x,y)'))
        if 'target' in sample:
            field_types.append(('target', None, 'viridis', 'Target'))
        
        # Homeostatic fields
        if 'alpha' in sample:
            field_types.append(('alpha', None, 'plasma', 'Stabilizer α'))
        
        n_rows = len(field_types)
        n_cols = n_show
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 3.5*n_rows))
        fig.suptitle('PDE Field Evolution During Training', fontsize=14, fontweight='bold')
        
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        if n_cols == 1:
            axes = axes.reshape(-1, 1)
        
        for row, (field_key, alt_key, cmap, title_base) in enumerate(field_types):
            for col, snap in enumerate(selected):
                ax = axes[row, col]
                
                # Get field data
                if field_key in snap:
                    field = snap[field_key]
                elif alt_key and alt_key in snap:
                    field = snap[alt_key]
                else:
                    ax.text(0.5, 0.5, 'N/A', ha='center', va='center', transform=ax.transAxes)
                    continue
                
                # Log scale for permeability
                if field_key == 'permeability':
                    field = np.log10(field + 1e-8)
                
                im = ax.imshow(field, cmap=cmap, aspect='auto')
                if row == 0:
                    ax.set_title(f'Step {snap["step"]}', fontsize=10)
                if col == 0:
                    ax.set_ylabel(title_base, fontsize=9)
                plt.colorbar(im, ax=ax, fraction=0.046)
                ax.set_xticks([])
                ax.set_yticks([])
        
        plt.tight_layout()
        plt.savefig(f'{self.report_dir}/12_pde_field_evolution.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("  ✓ 12_pde_field_evolution.png")
    
    def _plot_pde_stabilizer_gallery(self):
        """Gallery of stabilizer α patterns with input correlation."""
        snapshots = self.tracker.get_snapshots()
        if not snapshots:
            return
        
        sample = snapshots[0]
        has_alpha = 'alpha' in sample
        has_k = 'permeability' in sample
        has_boundary = 'boundary' in sample
        has_gamma = 'gamma' in sample
        
        if not has_alpha:
            return
        
        # Select snapshots
        n_show = min(6, len(snapshots))
        indices = np.linspace(0, len(snapshots)-1, n_show, dtype=int)
        selected = [snapshots[i] for i in indices]
        
        # Determine title based on problem type
        if has_k:
            title = 'Stabilizer α Evolution & Geological Intelligence'
            corr_xlabel = 'Permeability k'
            corr_label = 'ρ(k,α)'
        elif has_boundary:
            title = 'Stabilizer α Evolution & Boundary Awareness'
            corr_xlabel = 'Boundary Value'
            corr_label = 'ρ(BC,α)'
        else:
            title = 'Stabilizer α Evolution'
            corr_xlabel = 'α value'
            corr_label = ''
        
        fig = plt.figure(figsize=(16, 10))
        fig.suptitle(title, fontsize=14, fontweight='bold')
        
        n_cols = n_show
        
        for col, snap in enumerate(selected):
            # Top: alpha field
            ax1 = fig.add_subplot(2, n_cols, col + 1)
            alpha = snap['alpha']
            im = ax1.imshow(alpha, cmap='plasma', vmin=0, vmax=1, aspect='auto')
            ax1.set_title(f'Step {snap["step"]}', fontsize=10)
            if col == 0:
                ax1.set_ylabel('Stabilizer α', fontsize=10)
            plt.colorbar(im, ax=ax1, fraction=0.046)
            ax1.set_xticks([])
            ax1.set_yticks([])
            
            # Bottom: correlation plot
            ax2 = fig.add_subplot(2, n_cols, n_cols + col + 1)
            alpha_flat = alpha.flatten()
            gamma_val = snap.get('gamma', 0)
            
            if has_k:
                k = snap['permeability']
                k_flat = k.flatten()
                ax2.scatter(k_flat, alpha_flat, alpha=0.3, s=8, c='darkblue')
                ax2.set_xlabel(corr_xlabel)
                corr = np.corrcoef(k_flat, alpha_flat)[0, 1] if len(k_flat) > 1 else 0
                ax2.text(0.05, 0.95, f'{corr_label}={corr:.3f}\nγ={gamma_val:.3f}',
                        transform=ax2.transAxes, fontsize=9, va='top',
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
            elif has_boundary:
                boundary = snap['boundary']
                b_flat = boundary.flatten()
                # Filter for boundary points (non-zero boundary values)
                nonzero = np.abs(b_flat) > 1e-6
                if nonzero.sum() > 10:
                    ax2.scatter(b_flat[nonzero], alpha_flat[nonzero], alpha=0.4, s=10, c='darkgreen')
                    corr = np.corrcoef(b_flat[nonzero], alpha_flat[nonzero])[0, 1]
                else:
                    ax2.scatter(b_flat, alpha_flat, alpha=0.3, s=8, c='darkgreen')
                    corr = np.corrcoef(b_flat, alpha_flat)[0, 1] if len(b_flat) > 1 else 0
                ax2.set_xlabel(corr_xlabel)
                ax2.text(0.05, 0.95, f'{corr_label}={corr:.3f}\nγ={gamma_val:.3f}',
                        transform=ax2.transAxes, fontsize=9, va='top',
                        bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
            else:
                ax2.hist(alpha_flat, bins=30, color='purple', alpha=0.7)
                ax2.set_xlabel('α value')
                ax2.text(0.05, 0.95, f'γ={gamma_val:.3f}',
                        transform=ax2.transAxes, fontsize=9, va='top',
                        bbox=dict(boxstyle='round', facecolor='lavender', alpha=0.8))
            
            if col == 0:
                ax2.set_ylabel('α(x,y)')
            ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{self.report_dir}/13_pde_stabilizer_gallery.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("  ✓ 13_pde_stabilizer_gallery.png")
    
    def _plot_pde_error_maps(self):
        """Error maps showing where model struggles."""
        snapshots = self.tracker.get_snapshots()
        if not snapshots:
            return
        
        # Need prediction and target
        sample = snapshots[0]
        has_pred = 'prediction' in sample or 'pressure' in sample
        has_target = 'target' in sample
        
        if not (has_pred and has_target):
            return
        
        # Select snapshots
        n_show = min(4, len(snapshots))
        indices = np.linspace(0, len(snapshots)-1, n_show, dtype=int)
        selected = [snapshots[i] for i in indices]
        
        fig = plt.figure(figsize=(16, 12))
        fig.suptitle('Prediction Error Analysis', fontsize=14, fontweight='bold')
        
        for col, snap in enumerate(selected):
            pred = snap.get('prediction', snap.get('pressure'))
            target = snap['target']
            
            # Error map
            error = np.abs(pred - target)
            rel_error = error / (np.abs(target) + 1e-8)
            
            # Row 1: Prediction
            ax1 = fig.add_subplot(4, n_show, col + 1)
            im1 = ax1.imshow(pred, cmap='viridis', aspect='auto')
            ax1.set_title(f'Step {snap["step"]}', fontsize=10)
            if col == 0:
                ax1.set_ylabel('Prediction')
            plt.colorbar(im1, ax=ax1, fraction=0.046)
            ax1.set_xticks([]); ax1.set_yticks([])
            
            # Row 2: Target
            ax2 = fig.add_subplot(4, n_show, n_show + col + 1)
            im2 = ax2.imshow(target, cmap='viridis', aspect='auto')
            if col == 0:
                ax2.set_ylabel('Target')
            plt.colorbar(im2, ax=ax2, fraction=0.046)
            ax2.set_xticks([]); ax2.set_yticks([])
            
            # Row 3: Absolute Error
            ax3 = fig.add_subplot(4, n_show, 2*n_show + col + 1)
            im3 = ax3.imshow(error, cmap='hot', aspect='auto')
            if col == 0:
                ax3.set_ylabel('|Error|')
            plt.colorbar(im3, ax=ax3, fraction=0.046)
            ax3.set_xticks([]); ax3.set_yticks([])
            
            # MSE stats
            mse = np.mean(error**2)
            ax3.text(0.02, 0.98, f'MSE={mse:.4f}', transform=ax3.transAxes, fontsize=8,
                    va='top', bbox=dict(facecolor='white', alpha=0.8))
            
            # Row 4: Relative Error
            ax4 = fig.add_subplot(4, n_show, 3*n_show + col + 1)
            im4 = ax4.imshow(np.clip(rel_error, 0, 1), cmap='hot', vmin=0, vmax=0.5, aspect='auto')
            if col == 0:
                ax4.set_ylabel('Rel Error')
            plt.colorbar(im4, ax=ax4, fraction=0.046)
            ax4.set_xticks([]); ax4.set_yticks([])
        
        plt.tight_layout()
        plt.savefig(f'{self.report_dir}/14_pde_error_maps.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("  ✓ 14_pde_error_maps.png")
    
    def _plot_pde_correlation_evolution(self):
        """Track how input-α correlation evolves during training."""
        snapshots = self.tracker.get_snapshots()
        if not snapshots:
            return
        
        # Detect problem type
        sample = snapshots[0]
        has_k = 'permeability' in sample
        has_boundary = 'boundary' in sample
        
        # Compute correlation for each snapshot
        steps = []
        correlations = []
        gamma_vals = []
        alpha_means = []
        alpha_stds = []
        
        for snap in snapshots:
            if 'alpha' not in snap:
                continue
            
            alpha = snap['alpha']
            steps.append(snap['step'])
            alpha_means.append(np.mean(alpha))
            alpha_stds.append(np.std(alpha))
            
            if 'gamma' in snap:
                gamma_vals.append(snap['gamma'])
            
            # Compute correlation based on problem type
            if has_k and 'permeability' in snap:
                k = snap['permeability']
                corr = np.corrcoef(k.flatten(), alpha.flatten())[0, 1]
                correlations.append(corr)
            elif has_boundary and 'boundary' in snap:
                boundary = snap['boundary']
                b_flat = boundary.flatten()
                alpha_flat = alpha.flatten()
                # For boundary, focus on boundary points
                nonzero = np.abs(b_flat) > 1e-6
                if nonzero.sum() > 10:
                    corr = np.corrcoef(b_flat[nonzero], alpha_flat[nonzero])[0, 1]
                else:
                    corr = np.corrcoef(b_flat, alpha_flat)[0, 1] if len(b_flat) > 1 else 0
                correlations.append(corr)
            else:
                correlations.append(0)
        
        if not steps:
            return
        
        # Determine labels based on problem type
        if has_k:
            title = 'Geological Intelligence Evolution'
            corr_ylabel = 'Correlation ρ(k, α)'
            corr_title = 'Permeability-Stabilizer Correlation'
            success_msg = '✓ Strong Geological Learning'
        elif has_boundary:
            title = 'Boundary Awareness Evolution'
            corr_ylabel = 'Correlation ρ(BC, α)'
            corr_title = 'Boundary-Stabilizer Correlation'
            success_msg = '✓ Strong Boundary Awareness'
        else:
            title = 'Stabilizer Evolution'
            corr_ylabel = 'Correlation'
            corr_title = 'Input-Stabilizer Correlation'
            success_msg = '✓ Learned Adaptation'
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(title, fontsize=14, fontweight='bold')
        
        # Plot 1: Input-α Correlation over time
        ax1 = axes[0, 0]
        ax1.plot(steps, correlations, 'b-', linewidth=2, marker='o', markersize=4)
        ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax1.set_xlabel('Training Step')
        ax1.set_ylabel(corr_ylabel)
        ax1.set_title(corr_title)
        ax1.grid(True, alpha=0.3)
        
        # Add interpretation
        final_corr = correlations[-1] if correlations else 0
        if abs(final_corr) > 0.3:
            ax1.text(0.98, 0.02, success_msg, transform=ax1.transAxes,
                    fontsize=10, ha='right', va='bottom', color='green',
                    bbox=dict(facecolor='lightgreen', alpha=0.8))
        
        # Plot 2: α statistics over time
        ax2 = axes[0, 1]
        ax2.fill_between(steps, 
                        np.array(alpha_means) - np.array(alpha_stds),
                        np.array(alpha_means) + np.array(alpha_stds),
                        alpha=0.3, color='purple', label='±1σ')
        ax2.plot(steps, alpha_means, 'purple', linewidth=2, label='Mean α')
        ax2.set_xlabel('Training Step')
        ax2.set_ylabel('Stabilizer α')
        ax2.set_title('Stabilizer Statistics')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: γ evolution (if available)
        ax3 = axes[1, 0]
        if gamma_vals:
            ax3.plot(steps[:len(gamma_vals)], gamma_vals, 'g-', linewidth=2, marker='s', markersize=4)
            ax3.set_xlabel('Training Step')
            ax3.set_ylabel('Global Step Size γ')
            ax3.set_title('Spectral Controller Evolution')
            ax3.grid(True, alpha=0.3)
        else:
            ax3.text(0.5, 0.5, 'No γ data', ha='center', va='center', transform=ax3.transAxes, fontsize=12)
            ax3.set_title('Spectral Controller (not tracked)')
        
        # Plot 4: Combined phase-space
        ax4 = axes[1, 1]
        if gamma_vals and correlations:
            colors = np.linspace(0, 1, min(len(correlations), len(gamma_vals)))
            scatter = ax4.scatter(correlations[:len(gamma_vals)], gamma_vals, 
                                 c=colors, cmap='viridis', s=30, alpha=0.7)
            ax4.set_xlabel('k-α Correlation')
            ax4.set_ylabel('Global γ')
            ax4.set_title('Geological Intelligence Phase Space')
            plt.colorbar(scatter, ax=ax4, label='Training Progress')
            ax4.grid(True, alpha=0.3)
        else:
            ax4.text(0.5, 0.5, 'Insufficient data', ha='center', va='center', transform=ax4.transAxes)
        
        plt.tight_layout()
        plt.savefig(f'{self.report_dir}/15_pde_correlation_evolution.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("  ✓ 15_pde_correlation_evolution.png")


# ============================================================
# CONVENIENCE WRAPPER FOR QUICK INTEGRATION
# ============================================================

def create_reporter(experiment_name: str, base_dir: str = "reports"):
    """
    Quick setup for any training script.
    
    Returns:
        tracker: MetricsTracker to record metrics
        reporter: ReportManager to generate plots
    
    Usage:
        tracker, reporter = create_reporter("my_pde_deq")
        
        for step in range(max_steps):
            # ... training ...
            tracker.record(step=step, loss=loss.item(), phi=phi.item())
            
            if step % eval_every == 0:
                tracker.record_val(step, loss=val_loss, ppl=val_ppl)
        
        reporter.generate_all()
    """
    tracker = MetricsTracker()
    reporter = ReportManager(experiment_name, tracker, base_dir)
    return tracker, reporter


# ============================================================
# DEMO / SELF-TEST
# ============================================================

if __name__ == "__main__":
    print("DEQ Reports Module - Demo")
    print("=" * 50)
    
    # Create tracker and reporter
    tracker, reporter = create_reporter("demo_experiment")
    
    # Simulate training
    import random
    for step in range(500):
        loss = 5.0 * math.exp(-step / 200) + random.gauss(0, 0.1)
        phi = 0.8 + 0.15 * math.sin(step / 50) + random.gauss(0, 0.02)
        H = 5.5 + 0.3 * math.cos(step / 30) + random.gauss(0, 0.1)
        
        tracker.record(
            step=step,
            loss=loss,
            phi_raw=phi,
            phi_eff=math.log1p(phi),
            spectral_entropy=H,
            phi_penalty=max(0, phi - 0.95) ** 2,
            manifold_penalty=max(0, 5.4 - H) ** 2,
            gamma_phi=0.02 * math.exp(0.1 * random.gauss(0, 1)),
            gamma_manifold=0.05 * math.exp(0.1 * random.gauss(0, 1)),
            learning_rate=4e-4 * (0.95 ** (step // 100)),
            velocity=random.gauss(0, 0.1),
            acceleration=random.gauss(0, 0.05),
        )
        
        if step % 50 == 0 and step > 0:
            tracker.record_val(step, loss=loss + random.gauss(0.2, 0.1), ppl=math.exp(loss))
    
    # Generate reports
    reporter.generate_all()
    print("\n✅ Demo complete!")
