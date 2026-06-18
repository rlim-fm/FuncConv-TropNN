# FuncConv-TropNN
Functional Convergence Measures on Tropical Neural Networks

# Visualization Module
This module provides tools for visualizing convergence using 1-d and PCA-based 3D animations. It includes two modes of visualization:
1. **Static Manifold Mode**: Uses a fixed PCA basis from a single anchor epoch to visualize convergence across all epochs.
2. **Procrustes-Aligned Evolving Manifold Mode**: Computes a PCA basis for each epoch and aligns them using (orthogonal) Procrustes analysis to visualize the co-evolution of the ground truth manifold and predictions. 

## Quickstart
Simply run `train` followed by `visualization`.
