"""
Visualization script for analyzing functional convergence during training.
Reads from training_data.h5 and generates:
1. 1D functional convergence along x_1 axis
2. 3D PCA visualization of hidden state evolution
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
import h5py
import os


def load_training_data(h5_filename='out/training_data.h5'):
    """Load all training data from HDF5 file."""
    with h5py.File(h5_filename, 'r') as hf:
        # Load losses
        train_loss = hf['training/train_loss'][:]
        test_loss = hf['training/test_loss'][:]

        # Load test data
        x_test = hf['test/x_test'][:]
        y_test = hf['test/y_test'][:]
        predictions = hf['test/predictions'][:]  # [epochs, n_samples, 1]
        hidden_states = hf['test/hidden_states'][:]  # [epochs, n_samples, hidden_dim]

        # Load metadata
        metadata = dict(hf['metadata'].attrs)
        epochs = int(metadata['epochs'])  # Ensure it's a Python int, not numpy type
        
        # Squeeze predictions and y_test if needed (remove singleton dimensions)
        if predictions.ndim == 3 and predictions.shape[-1] == 1:
            predictions = predictions.squeeze(-1)
        if y_test.ndim == 2 and y_test.shape[-1] == 1:
            y_test = y_test.squeeze(-1)
        
    return {
        'train_loss': train_loss,
        'test_loss': test_loss,
        'x_test': x_test,
        'y_test': y_test,
        'predictions': predictions,
        'hidden_states': hidden_states,
        'epochs': epochs,
        'metadata': metadata
    }


def extract_1d_slice(x_test, y_test, predictions, hidden_states, axis=0, other_idx=None):
    """
    Extract a 1D slice of the data along the specified axis.

    Args:
        x_test: (n_samples, input_dim) - test inputs
        y_test: (n_samples, 1) - ground truth
        predictions: (epochs, n_samples, 1) - model predictions
        hidden_states: (epochs, n_samples, hidden_dim) - hidden states
        axis: which axis to vary (default: 0 for x_1)
        other_idx: sample index to use for other dimensions (if None, use mean)

    Returns:
        Sorted arrays for 1D visualization
    """
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
    preds_1d = predictions[:, indices]  # [epochs, n_selected_samples]
    hidden_1d = hidden_states[:, indices, :]  # [epochs, n_selected_samples, hidden_dim]
    
    return x_1d, y_1d, preds_1d, hidden_1d, indices


def plot_loss_history(train_loss, test_loss, output_path='loss_history.png'):
    """Plot training and test loss history."""
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
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"✓ Loss history saved to '{output_path}'")


def create_1d_animation(x_1d, y_1d, preds_1d, epochs, output_path='1d_convergence.gif'):
    """
    Create animation showing 1D functional convergence along x_1 axis.

    Args:
        x_1d: sorted x values along axis
        y_1d: ground truth y values
        preds_1d: predictions over epochs [epochs, n_samples]
        epochs: number of epochs
        output_path: where to save the GIF
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot ground truth
    ax.plot(x_1d, y_1d, 'k--', linewidth=3, label='Ground Truth', zorder=2)

    # Initialize prediction line
    line, = ax.plot([], [], 'b-', linewidth=2, label='Network Prediction', zorder=1)
    epoch_text = ax.text(0.05, 0.95, '', transform=ax.transAxes, fontsize=12,
                         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    # Set axis limits
    x_margin = (x_1d.max() - x_1d.min()) * 0.05
    y_min, y_max = y_1d.min(), y_1d.max()
    y_margin = (y_max - y_min) * 0.1

    ax.set_xlim(x_1d.min() - x_margin, x_1d.max() + x_margin)
    ax.set_ylim(y_min - y_margin, y_max + y_margin)
    ax.set_xlabel('$x_1$ (First Input Dimension)', fontsize=12)
    ax.set_ylabel('Output', fontsize=12)
    ax.set_title('Functional Convergence along $x_1$ Axis', fontsize=14)
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.legend(fontsize=11, loc='upper left')

    def init():
        line.set_data([], [])
        epoch_text.set_text('')
        return line, epoch_text

    def update(frame_idx):
        line.set_data(x_1d, preds_1d[frame_idx])
        epoch_text.set_text(f'Epoch: {frame_idx}')
        return line, epoch_text

    anim = FuncAnimation(fig, update, frames=epochs, init_func=init,
                        blit=True, interval=50)

    anim.save(output_path, writer='pillow', fps=15)
    plt.close(fig)
    print(f"✓ 1D convergence animation saved to '{output_path}'")


def create_pca_3d_animation(x_test, y_test, predictions, hidden_states, epochs,
                            output_path='pca_3d_convergence.gif'):
    """
    Create 3D animation showing convergence in PCA space of hidden states.

    Projects hidden states onto 2D PCA, with output as z-axis.

    Args:
        x_test: test inputs
        y_test: ground truth outputs
        predictions: predictions over epochs [epochs, n_samples, 1]
        hidden_states: hidden states over epochs [epochs, n_samples, hidden_dim]
        epochs: number of epochs
        output_path: where to save the GIF
    """
    # Fit PCA on all hidden states
    all_hidden_flat = hidden_states.reshape(-1, hidden_states.shape[-1])
    pca = PCA(n_components=2)
    pca.fit(all_hidden_flat)

    print(f"PCA explained variance ratio: {pca.explained_variance_ratio_}")

    # Transform the last epoch's hidden states to 2D
    hidden_last_2d = pca.transform(hidden_states[-1])
    pc1 = hidden_last_2d[:, 0]
    pc2 = hidden_last_2d[:, 1]
    y_flat = y_test.flatten()

    # Get ranges for all frames
    pc1_all = np.array([pca.transform(h)[:, 0] for h in hidden_states]).flatten()
    pc2_all = np.array([pca.transform(h)[:, 1] for h in hidden_states]).flatten()
    output_all = predictions.flatten()

    # Create 3D figure
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')

    # Plot ground truth using mesh
    ax.plot_trisurf(pc1, pc2, y_flat, cmap='viridis', alpha=0.8)

    # Initialize scatter plot for predictions
    scatter = ax.scatter([], [], [], c='blue', alpha=0.6, s=30, label='Network Predictions')
    epoch_text = ax.text2D(0.05, 0.95, '', transform=ax.transAxes, fontsize=12,
                          bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    # Set limits
    margin_pc1 = (pc1_all.max() - pc1_all.min()) * 0.1
    margin_pc2 = (pc2_all.max() - pc2_all.min()) * 0.1
    margin_out = (output_all.max() - output_all.min()) * 0.1

    ax.set_xlim(pc1_all.min() - margin_pc1, pc1_all.max() + margin_pc1)
    ax.set_ylim(pc2_all.min() - margin_pc2, pc2_all.max() + margin_pc2)
    ax.set_zlim(output_all.min() - margin_out, output_all.max() + margin_out)

    ax.set_xlabel('PC1', fontsize=11)
    ax.set_ylabel('PC2', fontsize=11)
    ax.set_zlabel('Output', fontsize=11)
    ax.set_title('Hidden State Evolution in PCA Space', fontsize=14)
    ax.legend(fontsize=10)
    ax.view_init(elev=20, azim=45)

    def init():
        scatter._offsets3d = ([], [], [])
        epoch_text.set_text('')
        return scatter, epoch_text

    def update(frame_idx):
        # Get predictions for this epoch
        output = predictions[frame_idx].flatten()
        scatter._offsets3d = (pc1, pc2, output)
        epoch_text.set_text(f'Epoch: {frame_idx}')
        return scatter, epoch_text

    anim = FuncAnimation(fig, update, frames=epochs, init_func=init,
                        blit=True, interval=50)

    anim.save(output_path, writer='pillow', fps=15)
    plt.close(fig)
    print(f"✓ PCA 3D animation saved to '{output_path}'")


def print_summary(data):
    """Print summary of loaded data."""
    print("\n" + "="*70)
    print("TRAINING DATA SUMMARY")
    print("="*70)
    print(f"Epochs: {data['epochs']}")
    print(f"Input dimension: {data['x_test'].shape[1]}")
    print(f"Hidden dimension: {data['hidden_states'].shape[2]}")
    print(f"Test samples: {data['x_test'].shape[0]}")
    print(f"\nLoss Statistics:")
    print(f"  Final train loss: {data['train_loss'][-1]:.6f}")
    print(f"  Final test loss:  {data['test_loss'][-1]:.6f}")
    print(f"  Best train loss:  {data['train_loss'].min():.6f} (epoch {data['train_loss'].argmin()})")
    print(f"  Best test loss:   {data['test_loss'].min():.6f} (epoch {data['test_loss'].argmin()})")
    print("\nOutput Statistics (Test Set):")
    print(f"  Ground truth - min: {data['y_test'].min():.4f}, max: {data['y_test'].max():.4f}")
    print(f"  Final pred   - min: {data['predictions'][-1].min():.4f}, max: {data['predictions'][-1].max():.4f}")
    print("="*70 + "\n")


def main():
    """Main visualization pipeline."""
    print("Loading training data...")
    data = load_training_data('out/training_data.h5')

    print_summary(data)

    # Create output directory if needed
    os.makedirs('visualizations', exist_ok=True)

    print("\nGenerating visualizations...")

    # 1. Loss history plot
    plot_loss_history(
        data['train_loss'],
        data['test_loss'],
        output_path='visualizations/loss_history.png'
    )


    # 2. 1D convergence along x_1 axis
    print("\nExtracting 1D slice along x₁ axis...")
    x_1d, y_1d, preds_1d, hidden_1d, indices = extract_1d_slice(
        data['x_test'], data['y_test'], data['predictions'],
        data['hidden_states'], axis=0
    )
    print(f"  Found {len(x_1d)} samples for 1D visualization")
    
    create_1d_animation(
        x_1d, y_1d, preds_1d,
        data['epochs'],
        output_path='visualizations/1d_convergence.gif'
    )

    # 3. PCA 3D visualization
    print("\nCreating PCA 3D visualization...")
    create_pca_3d_animation(
        data['x_test'], data['y_test'],
        data['predictions'], data['hidden_states'],
        data['epochs'],
        output_path='visualizations/pca_3d_convergence.gif'
    )

    print("\n" + "="*70)
    print("✓ All visualizations complete!")
    print("Output files:")
    print("  - visualizations/loss_history.png")
    print("  - visualizations/1d_convergence.gif")
    print("  - visualizations/pca_3d_convergence.gif")
    print("="*70 + "\n")


if __name__ == '__main__':
    main()
