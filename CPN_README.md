# Counter Propagation Network (CPN) Implementation

## Overview

This branch adds a **Counter Propagation Network (CPN)** implementation to complement the existing MLP architectures. CPNs are a type of neural network that combines unsupervised and supervised learning in a two-layer architecture.

## Architecture

A CPN consists of two layers:

1. **Kohonen Layer** (Competitive/Clustering Layer)
   - Uses unsupervised competitive learning
   - Winner-takes-all mechanism based on Euclidean distance
   - Clusters input patterns in feature space
   - Self-organizing with adaptive weight updates

2. **Grossberg Layer** (Output Layer)
   - Uses supervised learning
   - Maps winning Kohonen neurons to desired outputs
   - Trained via gradient descent

## Files Added

### `CPNx6.py`
Main CPN implementation following the same pattern as `MLPx6.py`:

- **7 predefined architectures** (`CPN_0` through `CPN_6`)
  - Varying numbers of Kohonen neurons: 8, 16, 32, 64, 128, 256, 512
  
- **Key Classes & Functions:**
  - `CPN`: Main model class implementing the two-layer architecture
  - `prepare_data()`: Data preparation with train/val/test splits
  - `train_model()`: Training loop with early stopping
  - `evaluate_model()`: Model evaluation and accuracy computation
  - `confusion_counts()`: Confusion matrix calculation
  - `predict()`: Single-point prediction
  - `plot_learning_curves()`: Training curve visualization
  - `plot_kohonen_neurons()`: Visualize Kohonen neuron positions in 2D space
  - `get_neuron_statistics()`: Analyze Kohonen neuron utilization

### `compare_mlp_cpn.py`
Interactive comparison tool for evaluating MLP vs CPN performance:

- **Comparison Functions:**
  - `compare_single_models()`: Compare one MLP vs one CPN architecture
  - `compare_all_architectures()`: Comprehensive comparison of all architectures
  - `plot_comparison()`: Visualization of learning curves
  - `plot_architecture_comparison()`: Multi-panel comparison plots

- **Interactive Menu:**
  - Option 1: Quick comparison (MLP_2 vs CPN_2)
  - Option 2: Compare all architectures
  - Option 3: Custom architecture selection
  - Option 4: Exit

## Usage Examples

### Basic CPN Training

```python
from CPNx6 import CPN, CPN_ARCHITECTURES, prepare_data, train_model, evaluate_model
from data import generate_intertwined_spirals, DATA_PARAMS

# Generate dataset
spirals = generate_intertwined_spirals(**DATA_PARAMS)

# Prepare data
train_loader, val_loader, test_loader = prepare_data(spirals)

# Create and train CPN with 32 Kohonen neurons
cpn = CPN(n_kohonen=32, input_size=2, output_size=1)
cpn, train_losses, val_losses, test_losses, train_time = train_model(
    cpn, train_loader, val_loader, test_loader, epochs=100
)

# Evaluate
accuracy = evaluate_model(cpn, test_loader)
print(f"CPN Accuracy: {accuracy:.4f}")
```

### Using the Comparison Tool

```bash
python compare_mlp_cpn.py
```

This launches an interactive menu for comparing architectures.

### Programmatic Comparison

```python
from compare_mlp_cpn import compare_single_models
from data import generate_intertwined_spirals, DATA_PARAMS

# Generate data
spirals = generate_intertwined_spirals(**DATA_PARAMS)

# Compare MLP_2 vs CPN_2
results = compare_single_models(
    spirals, 
    mlp_arch_name="MLP_2", 
    cpn_arch_name="CPN_2",
    epochs=100
)

# Results include accuracy, training time, confusion matrix, and more
print(f"MLP Accuracy: {results['mlp']['accuracy']:.4f}")
print(f"CPN Accuracy: {results['cpn']['accuracy']:.4f}")
```

### Visualizing Kohonen Neurons

```python
from CPNx6 import plot_kohonen_neurons, get_neuron_statistics

# After training a CPN model
plot_kohonen_neurons(cpn, x_range=(-1.5, 1.5), y_range=(-1.5, 1.5))

# Get neuron usage statistics
stats = get_neuron_statistics(cpn)
print(f"Active neurons: {stats['active_neurons']}/{cpn.n_kohonen}")
print(f"Utilization rate: {stats['utilization_rate']:.2%}")
```

## CPN vs MLP: Key Differences

| Aspect | MLP | CPN |
|--------|-----|-----|
| **Learning Type** | Fully supervised | Hybrid (unsupervised + supervised) |
| **Architecture** | Multiple hidden layers | Two layers (Kohonen + Grossberg) |
| **Weight Updates** | Backpropagation only | Competitive learning + backpropagation |
| **Feature Detection** | Implicit in hidden layers | Explicit in Kohonen layer |
| **Clustering** | Not inherent | Built-in via Kohonen layer |
| **Interpretability** | Lower | Higher (can visualize clusters) |

## CPN Architecture Options

```python
CPN_ARCHITECTURES = {
    "CPN_0": 8,      # 8 Kohonen neurons
    "CPN_1": 16,     # 16 Kohonen neurons
    "CPN_2": 32,     # 32 Kohonen neurons
    "CPN_3": 64,     # 64 Kohonen neurons
    "CPN_4": 128,    # 128 Kohonen neurons
    "CPN_5": 256,    # 256 Kohonen neurons
    "CPN_6": 512,    # 512 Kohonen neurons
}
```

## Comparison Metrics

The comparison tools evaluate models on:

- **Accuracy**: Classification accuracy on test set
- **Training Time**: Time to convergence (in seconds)
- **Epochs to Converge**: Number of epochs before early stopping
- **Final Test Loss**: Loss on test set at convergence
- **Confusion Matrix**: TN, FP, FN, TP counts
- **Precision & Recall**: Classification quality metrics

## Performance Considerations

### CPN Advantages:
- **Faster training** for certain problem types (especially with good cluster separation)
- **Better interpretability** through explicit clustering
- **Fewer parameters** compared to deep MLPs
- **Built-in feature extraction** via Kohonen layer

### CPN Limitations:
- **Fixed cluster count** must be chosen beforehand
- **Potential underutilization** of Kohonen neurons
- **Less flexible** than MLPs for complex decision boundaries
- **May require tuning** of two learning rates

### MLP Advantages:
- **More flexible** architecture (variable depth/width)
- **Better for complex patterns** with deep networks
- **Single learning rate** (simpler to tune)
- **Universal approximation** properties

### MLP Limitations:
- **More parameters** can lead to overfitting
- **Slower training** for deep networks
- **Less interpretable** decision process
- **No explicit clustering**

## Hyperparameters

### CPN-Specific:
- `n_kohonen`: Number of Kohonen neurons (8-512 in predefined architectures)
- `learning_rate_kohonen`: Learning rate for Kohonen layer updates (default: 0.1)
- `learning_rate_grossberg`: Learning rate for Grossberg layer (default: 0.01)

### Shared:
- `epochs`: Maximum training epochs (default: 100)
- `batch_size`: Mini-batch size (default: 32)
- `patience`: Early stopping patience (default: 10)
- `test_split`: Test set fraction (default: 0.2)
- `val_split`: Validation set fraction (default: 0.2)

## Future Enhancements

Potential improvements for the CPN implementation:

1. **Adaptive Kohonen Neurons**: Dynamically adjust number of neurons
2. **Neighborhood Learning**: Update neighbors of winning neuron (more like SOM)
3. **Learning Rate Decay**: Gradually decrease learning rates
4. **Multi-class Support**: Extend beyond binary classification
5. **GPU Acceleration**: Optimize for larger datasets
6. **Visualization Tools**: Interactive cluster exploration
7. **Ensemble Methods**: Combine multiple CPNs

## References

- Counter Propagation Networks were introduced by Robert Hecht-Nielsen (1987)
- Combines Kohonen Self-Organizing Maps with Grossberg's outstar learning
- Related to: Self-Organizing Maps (SOM), Radial Basis Function Networks (RBF)

## Testing

Run comparison on spiral dataset:

```bash
python compare_mlp_cpn.py
```

Select option 2 to run comprehensive comparison of all architectures.

## Branch Information

- **Branch Name**: `CPNnet`
- **Base Branch**: `master`
- **Status**: Development/Experimental
- **Purpose**: Add CPN as alternative to MLP for classification tasks
