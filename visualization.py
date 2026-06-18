"""
Visualization script for analyzing functional convergence during training.
Reads from training_data.h5 and generates:
1. 1D functional convergence along x_1 axis
2. 3D PCA visualization of hidden state evolution
"""
import warnings
from typing import Tuple, Optional

import math
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.linalg import orthogonal_procrustes
from sklearn.decomposition import PCA
import h5py
import os

from tqdm import trange
from util import ParameterizedCurve

class Visualizer:
    def __init__(self, data=None, metadata=None):
        self.data = data or {}
        self.metadata = metadata or {}

    def load_training_data(self, h5_filename='train_out/training_data.h5'):
        """Load all training data from HDF5 file."""
        with h5py.File(h5_filename, 'r') as hf:
            # Load losses
            train_loss = hf['logs/train_loss'][:]
            test_loss = hf['logs/test_loss'][:]

            # Load test data
            x_test = hf['logs/x_test'][:]
            y_test = hf['logs/y_test'][:]
            f_test = hf['logs/f_test'][:]  # [E, N]
            hidden_states = hf['logs/hidden_states'][:]  # [E, N, H]

            # Load metadata
            metadata = {key: hf['metadata'][key][()] for key in hf['metadata'].keys()}

        self.data = {
            'train_loss': train_loss,
            'test_loss': test_loss,
            'x_test': x_test.reshape(x_test.shape[0], -1),  # Flatten if needed
            'y_test': y_test.squeeze(), # [N,]
            'f_test': f_test.squeeze(), # [E, N]
            'hidden_states': hidden_states.reshape(hidden_states.shape[0], hidden_states.shape[1], -1),
        }
        self.metadata = metadata

    @staticmethod
    def from_processor_data(data, metadata):
        return Visualizer(data=data, metadata=metadata)

    def convergence_visualization_1d(self,
                                     line: Optional[ParameterizedCurve] = None,
                                     t_range: Optional[Tuple[float, float]] = None,
                                     output_path: Optional[str] = None,
                                     *,
                                     x_test: Optional[np.ndarray] = None,
                                     y_test: Optional[np.ndarray] = None,
                                     f_test: Optional[np.ndarray] = None):
        """
        Create animation showing 1D functional convergence along a parameterized line.

        Args:
            line: ParameterizedCurve object. If None, defaults to varying the first input dimension.
            t_range: (t_min, t_max) parameter x_range. If None, defaults to data x_range on first axis.
            output_path: where to save the GIF
            x_test: (n_samples, input_dim) test inputs for determining line and ground truth
            y_test: (n_samples,) ground truth outputs for the test inputs
            f_test: (epochs, n_samples) model predictions over epochs for the test inputs

        Returns:
            FuncAnimation object for the 1D convergence animation
        """
        input_dim = math.prod(self.metadata['data_dim'])
        if x_test is None:
            x_test = self.data['x_test']
        if y_test is None:
            y_test = self.data['y_test']
        if f_test is None:
            f_test = self.data['f_test']
        epochs = int(self.metadata.get('epochs', len(f_test)))

        # Default line: vary along first input dimension only
        if line is None:
            base = np.mean(x_test, axis=0)
            line = ParameterizedCurve.axis_curve(input_dim, axis=0)

        # Default parameter x_range: match data x_range
        if t_range is None:
            t_range = (np.min(x_test[:, 0]), np.max(x_test[:, 0]))

        # Generate line points for visualization
        n_line_points = 256
        t_values = np.linspace(t_range[0], t_range[1], n_line_points)
        line_points = line(t_values)  # [n_line_points, input_dim]

        # Evaluate ground truth and f_test along the line
        with torch.no_grad():
            line_points_torch = torch.from_numpy(line_points).float()
            if line_points_torch.dim() == 1:
                line_points_torch = line_points_torch.unsqueeze(-1)
            ground_truth_line = y_test  # This will need to match line evaluation
            pred_line = np.array([f_test[epoch] for epoch in range(epochs)])

        # For simplicity, extract 1D slice along first axis for ground truth
        x_1d, y_1d, preds_1d, _, _ = self._extract_1d_slice(axis=0)

        # Create animation using the extracted 1D data
        fig, ax = plt.subplots(figsize=(12, 7))

        # Plot ground truth analytical line
        ax.plot(x_1d, y_1d, 'k--', linewidth=3, label='Ground Truth', zorder=2)

        # Initialize prediction line
        line_anim, = ax.plot([], [], 'b-', linewidth=2, label='Network Prediction', zorder=1)
        epoch_text = ax.text(0.05, 0.95, '', transform=ax.transAxes, fontsize=12,
                            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

        # Set axis limits
        x_margin = (x_1d.max() - x_1d.min()) * 0.05
        y_min, y_max = float(min(y_1d.min(), preds_1d.min())), float(max(y_1d.max(), preds_1d.max()))
        y_margin = (y_max - y_min) * 0.1

        ax.set_xlim(x_1d.min() - x_margin, x_1d.max() + x_margin)
        ax.set_ylim(y_min - y_margin, y_max + y_margin)
        ax.set_xlabel(r'$t$, fontsize=12')
        ax.set_ylabel('Output', fontsize=12)
        ax.set_title('Functional Convergence along Parameterized Line', fontsize=14)
        ax.grid(True, linestyle='--', alpha=0.4)
        ax.legend(fontsize=11, loc='upper right')

        def init():
            line_anim.set_data([], [])
            epoch_text.set_text('')
            return line_anim, epoch_text

        def update(frame_idx):
            line_anim.set_data(x_1d, preds_1d[frame_idx])
            epoch_text.set_text(f'Epoch: {frame_idx}')
            return line_anim, epoch_text

        anim = FuncAnimation(fig, update, frames=epochs, init_func=init,
                            blit=True, interval=50)
        if output_path is not None:
            anim.save(output_path, writer='pillow', fps=15)
            print(f"✓ 1D convergence animation saved to '{output_path}'")
        
        plt.close(fig)
        return anim

    def hidden_layer_visualization(self,
                                   pca_epoch: int | str=-1,
                                   output_path='visualizations/hidden_layer_pca.gif', 
                                   *,
                                   y_test: Optional[np.ndarray] = None,
                                   f_test: Optional[np.ndarray] = None,
                                   hidden_states: Optional[np.ndarray] = None):
        """
        Create 3D animation showing convergence in PCA space of hidden states.

        Supports two modes:
        1. Numeric (default): Fit PCA on anchor epoch, create static manifold for y_test
        2. 'all' mode: Fit PCA on each epoch individually with Procrustes alignment

        Projects hidden states onto 2D PCA, with output as z-axis.

        Args:
            y_test: ground truth outputs [N, 1]
            f_test: f_test over epochs [E, N, 1]
            hidden_states: hidden states over epochs [E, N, hidden_dim]
            pca_epoch: anchor epoch for PCA fitting. If numeric (default -1), use single fixed PCA.
                      If 'all', fit PCA per epoch with Procrustes alignment for smoothness.
            output_path: where to save the GIF

        Returns:
            FuncAnimation object for the PCA 3D convergence animation
        """
        if y_test is None:
            y_test = self.data['y_test']
        if f_test is None:
            f_test = self.data['f_test']
        if hidden_states is None:
            hidden_states = self.data['hidden_states']
        
        assert f_test.shape[
                   1:] == y_test.shape, f"preds shape {f_test.shape}[1:] must match y_test shape {y_test.shape} on dimensions"
        epochs = hidden_states.shape[0]
        n_samples = hidden_states.shape[1]
        y_flat = y_test.flatten()

        print(
            f"Creating PCA 3D animation in mode: {'all (Procrustes-aligned)' if pca_epoch == 'all' else f'anchor epoch={pca_epoch}'}")

        if pca_epoch == "all":
            # Mode 2: Fit PCA on each epoch with Procrustes alignment
            pca_list = []
            hidden_2d_list = []

            # Fit PCA for each epoch
            for epoch_idx in range(epochs):
                pca = PCA(n_components=2)
                pca.fit(hidden_states[epoch_idx, :, :])
                hidden_2d = pca.transform(hidden_states[epoch_idx, :, :])  # [N, 2]
                pca_list.append(pca)
                hidden_2d_list.append(hidden_2d)

            # Apply Procrustes alignment for smoothness
            for epoch_idx in range(1, epochs):
                # Align current epoch to previous epoch
                R, _ = orthogonal_procrustes(hidden_2d_list[epoch_idx], hidden_2d_list[epoch_idx - 1])
                hidden_2d_list[epoch_idx] @= R

            # Extract PC coordinates for all epochs
            pc1_all_frames = np.array([h[:, 0] for h in hidden_2d_list])  # [E, N]
            pc2_all_frames = np.array([h[:, 1] for h in hidden_2d_list])  # [E, N]

            # Initial manifold: use PCA from first epoch for z-axis reference
            pca_ref = pca_list[0]
            print(f"PCA explained variance ratio (epoch 1): {pca_ref.explained_variance_ratio_}")

            # Anchor points for manifold surface
            pc1_anchor = pc1_all_frames[0]
            pc2_anchor = pc2_all_frames[0]

        else:
            # Mode 1 (default): Single fixed PCA on anchor epoch
            pca = PCA(n_components=2)
            pca.fit(hidden_states[pca_epoch, :, :])
            print(f"PCA explained variance ratio (anchor epoch {pca_epoch}): {pca.explained_variance_ratio_}")

            # Transform all epochs using single PCA basis
            pc1_all_frames = np.array(
                [pca.transform(hidden_states[epoch_idx, :, :])[:, 0] for epoch_idx in range(epochs)])
            pc2_all_frames = np.array(
                [pca.transform(hidden_states[epoch_idx, :, :])[:, 1] for epoch_idx in range(epochs)])

            # Use anchor epoch for static manifold
            hidden_anchor_2d = pca.transform(hidden_states[pca_epoch, :, :])
            pc1_anchor = hidden_anchor_2d[:, 0]
            pc2_anchor = hidden_anchor_2d[:, 1]

        # ===== Create 3D visualization =====
        fig = plt.figure(figsize=(12, 9))
        ax = fig.add_subplot(111, projection='3d')
        surf_plot = ax.plot_trisurf(pc1_anchor, pc2_anchor, y_flat, cmap='viridis', alpha=0.3,
                                    label='Ground Truth Surface')

        # Initialize scatter plot for predictions
        scatter = ax.scatter([], [], [], c='black', alpha=0.8, s=30, label='Network Predictions')
        epoch_text = ax.text2D(0.05, 0.95, '', transform=ax.transAxes, fontsize=12,
                               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

        # Compute global ranges for consistent axis limits
        pc1_all = pc1_all_frames.flatten()
        pc2_all = pc2_all_frames.flatten()
        output_all = f_test.flatten()

        margin_pc1 = (pc1_all.max() - pc1_all.min()) * 0.1
        margin_pc2 = (pc2_all.max() - pc2_all.min()) * 0.1
        margin_out = (output_all.max() - output_all.min()) * 0.1
        x_lim = pc1_all.min() - margin_pc1, pc1_all.max() + margin_pc1
        y_lim = pc2_all.min() - margin_pc2, pc2_all.max() + margin_pc2
        z_lim = output_all.min() - margin_out, output_all.max() + margin_out

        def set_static():
            ax.set_xlim(x_lim)
            ax.set_ylim(y_lim)
            ax.set_zlim(z_lim)

            ax.set_xlabel('PC1', fontsize=11)
            ax.set_ylabel('PC2', fontsize=11)
            ax.set_zlabel('Output', fontsize=11)
            mode_str = "All Epochs (Procrustes)" if pca_epoch == "all" else f"Anchor Epoch {pca_epoch}"
            ax.set_title(f'Hidden State Evolution in PCA Space [{mode_str}]', fontsize=14)
            ax.legend(fontsize=10, loc='upper right')
            ax.view_init(elev=20, azim=45)

        set_static()

        def init():
            scatter._offsets3d = ([], [], [])
            epoch_text.set_text('')
            return scatter, epoch_text

        def update(frame_idx):
            # Get f_test and PCA coordinates for this epoch
            pc1_frame = pc1_all_frames[frame_idx]
            pc2_frame = pc2_all_frames[frame_idx]
            if pca_epoch == "all":
                nonlocal surf_plot
                surf_plot.remove()
                surf_plot = ax.plot_trisurf(pc1_frame, pc2_frame, y_flat, cmap='viridis', alpha=0.3,
                                            label='Ground Truth Surface')
                set_static()
            output = f_test[frame_idx].flatten()
            scatter._offsets3d = (pc1_frame, pc2_frame, output)
            epoch_text.set_text(f'Epoch: {frame_idx}')
            return scatter, epoch_text

        anim = FuncAnimation(fig, update, frames=epochs, init_func=init,
                             blit=True, interval=50)

        anim.save(output_path, writer='pillow', fps=15)
        plt.close(fig)
        print(f"✓ PCA 3D animation saved to '{output_path}'")
        return anim


    def _extract_1d_slice(self,
                          axis=0,
                          other_idx=None,
                          hidden_states: Optional[np.ndarray] = None,
                          x_test: Optional[np.ndarray] = None,
                          y_test: Optional[np.ndarray] = None,
                          f_test: Optional[np.ndarray] = None,
                          **kwargs):
        """
        Extract a 1D slice of the data along the specified axis.
        
        Args:
            x_test: (n_samples, input_dim) - test inputs
            y_test: (n_samples, 1) - ground truth
            f_test: (epochs, n_samples, 1) - model f_test
            hidden_states: (epochs, n_samples, hidden_dim) - hidden states
            axis: which axis to vary (default: 0 for x_1)
            other_idx: sample index to use for other dimensions (if None, use mean)
        
        Returns:
            Sorted arrays for 1D visualization
        """
        if x_test is None:
            x_test = self.data['x_test']
        if y_test is None:
            y_test = self.data['y_test']
        if f_test is None:
            f_test = self.data['f_test']
        if hidden_states is None:
            hidden_states = self.data['hidden_states']
        n_samples = x_test.shape[0]
        
        # Find samples that vary along the specified axis
        # Use a slice where other dimensions are approximately constant
        if other_idx is None:
            # Create a slice by selecting samples near the median of other dimensions
            mask = np.ones(n_samples, dtype=bool)
            for d in range(x_test.shape[1]):
                if d != axis:
                    median_val = np.median(x_test[:, d])
                    tolerance = 0.2 * (np.max(x_test[:, d]) - np.min(x_test[:, d]))
                    mask &= np.abs(x_test[:, d] - median_val) < tolerance
            indices = np.where(mask)[0]
        else:
            indices = np.arange(n_samples)
        
        if len(indices) < 2:
            print(f"Warning: Only {len(indices)} samples found for 1D slice")
            # Fallback: just take all samples and sort by axis
            indices = np.arange(n_samples)
        
        # Sort by the specified axis
        sort_idx = np.argsort(x_test[indices, axis])
        indices = indices[sort_idx]
        
        x_1d = x_test[indices, axis]
        y_1d = y_test[indices]
        preds_1d = f_test[:, indices]  # [epochs, n_selected_samples]
        hidden_1d = hidden_states[:, indices, :]  # [epochs, n_selected_samples, hidden_dim]
        
        return x_1d, y_1d, preds_1d, hidden_1d, indices


    def plot_loss_history(self, 
                          output_path: Optional[str] = None,
                          *,
                          train_loss: Optional[np.ndarray] = None,
                          test_loss: Optional[np.ndarray] = None):
        """Plot training and test loss history."""
        if train_loss is None:
            train_loss = self.data['train_loss']
        if test_loss is None:
            test_loss = self.data['test_loss']
        plt.figure(figsize=(10, 6))
        plt.plot(train_loss, linewidth=2, label='Train Loss', alpha=0.8)
        plt.plot(test_loss, linewidth=2, label='Test Loss', alpha=0.8)
        plt.title('Training Loss History', fontsize=14)
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('MSE Loss', fontsize=12)
        plt.yscale('log')
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend(fontsize=11)
        plt.tight_layout()
        if output_path is not None:
            plt.savefig(output_path, dpi=150)
            plt.close()
            print(f"✓ Loss history saved to '{output_path}'")
        return plt


    def print_summary(self):
        """Print training summary from logged data."""
        preds_shape = np.array(self.data['f_test']).shape
        hidden_shape = np.array(self.data['hidden_states']).shape
    
        print("\n" + "=" * 75)
        print("TRAINING SUMMARY (from logged data)")
        print("=" * 75)
        print(f"\n[Loss Statistics]")
        print(f"  Final train loss: {self.data['train_loss'][-1]:.6f}")
        print(f"  Final test loss:  {self.data['test_loss'][-1]:.6f}")
        print(f"  Best train loss:  {np.min(self.data['train_loss']):.6f} (epoch {np.argmin(self.data['train_loss'])})")
        print(f"  Best test loss:   {np.min(self.data['test_loss']):.6f} (epoch {np.argmin(self.data['test_loss'])})")
        print(f"\n[HDF5 File Structure]")
        print(f"  ├── logs/")
        print(f"  │   ├── x_train: {self.data['x_train'].shape}")
        print(f"  │   ├── y_train: {self.data['y_train'].shape}")
        print(f"  │   ├── x_test: {self.data['x_test'].shape}")
        print(f"  │   ├── y_test: {self.data['y_test'].shape}")
        print(f"  │   ├── train_loss: {len(self.data['train_loss'])} entries")
        print(f"  │   ├── test_loss: {len(self.data['test_loss'])} entries")
        print(f"  │   ├── f_test: {preds_shape} entries")
        print(f"  │   └── hidden_states: {hidden_shape} entries")
        print(f"  └── metadata/")
        print(f"      ├── x_range: {self.metadata['x_range']}")
        print(f"      ├── data_dim: {self.metadata['data_dim']}")
        print(f"      ├── N: {self.metadata['N']}")
        print(f"      ├── ground_truth: {self.metadata['ground_truth']}")
        print(f"      ├── model: {self.metadata['model']}")
        print(f"      ├── optimizer: {self.metadata['optimizer']}")
        print(f"      ├── criterion: {self.metadata['criterion']}")
        print(f"      ├── scheduler: {self.metadata['scheduler']}")
        print(f"      └── epochs: {self.metadata['epochs']}")
        print("=" * 75 + "\n")

    def visualize_full(self, name):
        self.print_summary()

        # Create output directory if needed
        os.makedirs('visualizations', exist_ok=True)
        print("\nGenerating visualizations...")

        # 1. Loss history plot
        print("\nCreating loss history plot...")
        self.plot_loss_history(
            output_path=f'visualizations/topk-sum/{name}_loss_history.png'
        )

        # 2. 1D convergence with default axis-aligned line (along x_1)
        print("\nCreating 1D convergence animation...")
        self.convergence_visualization_1d(
            line=None,  # Uses default: axis-aligned along x_1
            t_range=None,  # Uses data x_range
            output_path=f'visualizations/topk-sum/{name}_1d_convergence.gif'
        )

        # 3. PCA 3D visualization - Mode 1: Default (anchor epoch)
        print("\nCreating PCA 3D visualization (anchor epoch mode)...")
        self.hidden_layer_visualization(
            pca_epoch=-1,
            output_path=f'visualizations/topk-sum/{name}_pca_3d_convergence.gif'
        )

        # 3b. PCA 3D visualization - Mode 2: All epochs with Procrustes alignment
        print("\nCreating PCA 3D visualization (all epochs + Procrustes mode)...")
        self.hidden_layer_visualization(
            pca_epoch='all',
            output_path=f'visualizations/topk-sum/{name}_pca_3d_convergence_procrustes.gif'
        )

        print("\n" + "="*70)
        print("✓ All visualizations complete!")
        print("Output files:")
        print(f"  - Loss history: visualizations/topk-sum/{name}_loss_history.png")
        print(f"  - 1D convergence: visualizations/topk-sum/{name}_1d_convergence.gif")
        print(f"  - PCA 3D convergence (anchor epoch): visualizations/topk-sum/{name}_pca_3d_convergence.gif")
        print(f"  - PCA 3D convergence (all epochs + Procrustes): visualizations/topk-sum/{name}_pca_3d_convergence_procrustes.gif")
        print("="*70 + "\n")

def main():
    """Main visualization pipeline"""
    train_filename = "MHA_training_data"
    print("Loading training data...")
    visualizer = Visualizer()
    visualizer.load_training_data(f'train_out/{train_filename}.h5')
    visualizer.visualize_full(train_filename)

if __name__ == '__main__':
    main()
