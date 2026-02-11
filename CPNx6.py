import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
import time


# ============================
# CPN Architectures (easily modifiable)
# ============================

CPN_ARCHITECTURES = {
    "CPN_0": 8,      # 8 Kohonen neurons
    "CPN_1": 16,     # 16 Kohonen neurons
    "CPN_2": 32,     # 32 Kohonen neurons
    "CPN_3": 64,     # 64 Kohonen neurons
    "CPN_4": 128,    # 128 Kohonen neurons
    "CPN_5": 256,    # 256 Kohonen neurons
    "CPN_6": 512,    # 512 Kohonen neurons
}


class CPN(nn.Module):
    """
    Counter Propagation Network (CPN) for binary classification.
    
    A CPN consists of two layers:
    1. Kohonen layer (competitive/clustering layer) - unsupervised learning
    2. Grossberg layer (output layer) - supervised learning
    
    Architecture:
    - Input -> Kohonen (competitive) -> Grossberg (output)
    - Kohonen layer uses winner-takes-all with Euclidean distance
    - Grossberg layer maps winning neuron to output
    
    Parameters:
    -----------
    n_kohonen : int
        Number of neurons in the Kohonen (competitive) layer
    input_size : int
        Input dimension (default 2 for 2D coordinates)
    output_size : int
        Output dimension (default 1 for binary classification)
    learning_rate_kohonen : float
        Learning rate for Kohonen layer weight updates (default 0.1)
    learning_rate_grossberg : float
        Learning rate for Grossberg layer (default 0.01)
    """
    def __init__(self, n_kohonen, input_size=2, output_size=1, 
                 learning_rate_kohonen=0.1, learning_rate_grossberg=0.01):
        super(CPN, self).__init__()
        
        if n_kohonen <= 0:
            raise ValueError("n_kohonen must be positive")
        
        self.n_kohonen = n_kohonen
        self.input_size = input_size
        self.output_size = output_size
        self.lr_kohonen = learning_rate_kohonen
        self.lr_grossberg = learning_rate_grossberg
        
        # Kohonen layer weights (competitive layer)
        # Initialize with small random values normalized to unit length
        self.kohonen_weights = nn.Parameter(torch.randn(n_kohonen, input_size))
        # Normalize initial weights
        with torch.no_grad():
            self.kohonen_weights.data = nn.functional.normalize(self.kohonen_weights.data, dim=1)
        
        # Grossberg layer (output layer)
        self.grossberg = nn.Linear(n_kohonen, output_size)
        self.sigmoid = nn.Sigmoid()
        
        # Track which Kohonen neurons have been activated (for adaptive learning)
        self.neuron_counts = torch.zeros(n_kohonen)
    
    def find_winner(self, x):
        """
        Find the winning Kohonen neuron using Euclidean distance.
        
        Parameters:
        -----------
        x : torch.Tensor
            Input batch of shape (batch_size, input_size)
        
        Returns:
        --------
        winners : torch.Tensor
            Index of winning neuron for each sample (batch_size,)
        distances : torch.Tensor
            Distances to all neurons (batch_size, n_kohonen)
        """
        # Normalize input
        x_norm = nn.functional.normalize(x, dim=1)
        
        # Compute Euclidean distances to all Kohonen neurons
        # distances[i, j] = ||x[i] - w[j]||^2
        distances = torch.cdist(x_norm.unsqueeze(1), 
                               self.kohonen_weights.unsqueeze(0)).squeeze(1)
        
        # Find winning neuron (minimum distance)
        winners = torch.argmin(distances, dim=1)
        
        return winners, distances
    
    def kohonen_forward(self, x):
        """
        Forward pass through Kohonen layer.
        Returns one-hot encoding of winning neurons.
        """
        winners, _ = self.find_winner(x)
        
        # Create one-hot encoding
        kohonen_output = torch.zeros(x.size(0), self.n_kohonen, device=x.device)
        kohonen_output.scatter_(1, winners.unsqueeze(1), 1.0)
        
        return kohonen_output, winners
    
    def forward(self, x):
        """
        Full forward pass through the CPN.
        """
        # Kohonen layer (competitive)
        kohonen_output, winners = self.kohonen_forward(x)
        
        # Grossberg layer (output)
        output = self.grossberg(kohonen_output)
        output = self.sigmoid(output)
        
        return output, winners
    
    def update_kohonen(self, x, winners):
        """
        Update Kohonen layer weights using competitive learning rule.
        Only the winning neuron's weights are updated.
        
        w_winner = w_winner + lr * (x - w_winner)
        """
        with torch.no_grad():
            x_norm = nn.functional.normalize(x, dim=1)
            
            for i, winner_idx in enumerate(winners):
                # Update winning neuron
                self.kohonen_weights[winner_idx] += self.lr_kohonen * (
                    x_norm[i] - self.kohonen_weights[winner_idx]
                )
                
                # Track neuron activation count
                self.neuron_counts[winner_idx] += 1
            
            # Normalize weights to maintain unit length
            self.kohonen_weights.data = nn.functional.normalize(
                self.kohonen_weights.data, dim=1
            )


def prepare_data(spirals_data, test_split=0.2, val_split=0.2, batch_size=32):
    """
    Prepare data for training with 3-way split: train/validation/test.

    Parameters:
    -----------
    spirals_data : list of tuples
        Output from generate_intertwined_spirals()
    test_split : float
        Fraction of data for testing (default 0.2, i.e., 20%)
    val_split : float
        Fraction of REMAINING data for validation after test split (default 0.2, i.e., 20% of 80% = 16%)
        Results in: 64% train, 16% validation, 20% test
    batch_size : int
        Batch size for DataLoader

    Returns:
    --------
    train_loader, val_loader, test_loader : DataLoader objects
    """
    # Extract x, y, and labels
    x_data = np.array([[x, y] for x, y, _ in spirals_data], dtype=np.float32)
    y_data = np.array([label for _, _, label in spirals_data], dtype=np.float32).reshape(-1, 1)

    # Convert to tensors
    x_tensor = torch.from_numpy(x_data)
    y_tensor = torch.from_numpy(y_data)

    # Split into train+val and test
    n_samples = len(spirals_data)
    n_test = int(n_samples * test_split)
    n_trainval = n_samples - n_test

    indices = np.random.permutation(n_samples)
    trainval_indices = indices[:n_trainval]
    test_indices = indices[n_trainval:]

    # Further split trainval into train and validation
    n_val = int(n_trainval * val_split)
    n_train = n_trainval - n_val

    train_indices = trainval_indices[:n_train]
    val_indices = trainval_indices[n_train:]

    train_dataset = TensorDataset(x_tensor[train_indices], y_tensor[train_indices])
    val_dataset = TensorDataset(x_tensor[val_indices], y_tensor[val_indices])
    test_dataset = TensorDataset(x_tensor[test_indices], y_tensor[test_indices])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


def train_model(model, train_loader, val_loader, test_loader, epochs=100, 
                lr_grossberg=0.001, patience=10, verbose=True, optimizer_type="adam"):
    """
    Train a CPN model with validation set for early stopping.
    
    The CPN training has two phases:
    1. Kohonen layer updates (competitive learning) - custom update rule
    2. Grossberg layer updates (supervised learning) - gradient descent

    Parameters:
    -----------
    model : CPN
        The CPN model to train
    train_loader : DataLoader
        Training data loader (64% of data)
    val_loader : DataLoader
        Validation data loader (16% of data, used for early stopping)
    test_loader : DataLoader
        Testing data loader (20% of data, for final evaluation)
    epochs : int
        Number of training epochs
    lr_grossberg : float
        Learning rate for Grossberg layer (Kohonen lr is set in model)
    patience : int
        Early stopping patience (epochs without improvement before stopping)
    verbose : bool
        Print progress
    optimizer_type : str
        "adam" or "sgd" for Grossberg layer optimization

    Returns:
    --------
    model : trained CPN model
    train_losses : list of training losses
    val_losses : list of validation losses
    test_losses : list of test losses
    train_time_ms : float
        Elapsed time used for training in milliseconds
    """
    criterion = nn.BCELoss()
    
    # Optimizer only for Grossberg layer
    if optimizer_type == "adam":
        optimizer = optim.Adam(model.grossberg.parameters(), lr=lr_grossberg)
    elif optimizer_type == "sgd":
        optimizer = optim.SGD(model.grossberg.parameters(), lr=lr_grossberg)
    else:
        raise ValueError("optimizer_type must be 'adam' or 'sgd'")

    train_losses = []
    val_losses = []
    test_losses = []

    # Early stopping parameters (using validation set)
    best_val_loss = float('inf')
    best_model_state = None
    best_epoch = 0
    patience_counter = 0

    start_time = time.perf_counter()

    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        
        for x_batch, y_batch in train_loader:
            # Forward pass
            outputs, winners = model(x_batch)
            
            # Update Kohonen layer (competitive learning)
            model.update_kohonen(x_batch, winners)
            
            # Update Grossberg layer (supervised learning)
            optimizer.zero_grad()
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()

        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        # Validation phase (for early stopping)
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                outputs, _ = model(x_batch)
                loss = criterion(outputs, y_batch)
                val_loss += loss.item()

        val_loss /= len(val_loader)
        val_losses.append(val_loss)

        # Early stopping check based on validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
            patience_counter = 0
        else:
            patience_counter += 1

        # Test phase (final evaluation)
        test_loss = 0.0
        with torch.no_grad():
            for x_batch, y_batch in test_loader:
                outputs, _ = model(x_batch)
                loss = criterion(outputs, y_batch)
                test_loss += loss.item()

        test_loss /= len(test_loader)
        test_losses.append(test_loss)

        if verbose and (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}/{epochs} - Train: {train_loss:.4f}, Val: {val_loss:.4f}, Test: {test_loss:.4f}")

        # Stop if validation loss hasn't improved
        if patience_counter >= patience:
            if verbose:
                print(f"Early stopping at epoch {epoch + 1} (best epoch: {best_epoch + 1}, Val Loss: {best_val_loss:.4f})")
            break

    train_time_ms = (time.perf_counter() - start_time) * 1000.0

    if verbose:
        train_time_s = train_time_ms / 1000.0
        print(f"Training time: {train_time_s:.2f} s")

    # Restore best model state
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        # Truncate loss lists to best epoch
        train_losses = train_losses[:best_epoch + 1]
        val_losses = val_losses[:best_epoch + 1]
        test_losses = test_losses[:best_epoch + 1]

    return model, train_losses, val_losses, test_losses, train_time_ms


def confusion_counts(model, test_loader):
    """Return confusion counts (TN, FP, FN, TP) on the test set."""
    model.eval()
    tn = fp = fn = tp = 0
    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            outputs, _ = model(x_batch)
            predicted = (outputs > 0.5).float()
            tn += ((predicted == 0) & (y_batch == 0)).sum().item()
            fp += ((predicted == 1) & (y_batch == 0)).sum().item()
            fn += ((predicted == 0) & (y_batch == 1)).sum().item()
            tp += ((predicted == 1) & (y_batch == 1)).sum().item()
    return tn, fp, fn, tp


def predict(model, x, y):
    """
    Predict class for a single point.

    Parameters:
    -----------
    model : CPN
        Trained CPN model
    x, y : float
        Coordinates of the point

    Returns:
    --------
    prediction : 0 (C1) or 1 (C2)
    confidence : float between 0 and 1
    """
    model.eval()
    with torch.no_grad():
        input_tensor = torch.tensor([[x, y]], dtype=torch.float32)
        output, _ = model(input_tensor)
        output = output.item()

    prediction = 1 if output > 0.5 else 0
    confidence = output if prediction == 1 else (1 - output)

    return prediction, confidence


def evaluate_model(model, test_loader):
    """
    Evaluate CPN model on test set and return accuracy.

    Parameters:
    -----------
    model : CPN
        Trained CPN model
    test_loader : DataLoader
        Testing data loader

    Returns:
    --------
    accuracy : float between 0 and 1
    """
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            outputs, _ = model(x_batch)
            predicted = (outputs > 0.5).float()
            total += y_batch.size(0)
            correct += (predicted == y_batch).sum().item()

    return correct / total


def plot_learning_curves(train_histories, title="CPN Learning Curves", fig=None):
    """
    Plot learning curves (loss vs epochs) for one or more CPN models.

    Parameters:
    -----------
    train_histories : dict
        {model_name: {"train_losses": [...], "test_losses": [...]}}
    title : str
        Plot title
    """
    if not train_histories:
        return None

    if fig is None or not plt.fignum_exists(fig.number):
        fig = plt.figure(figsize=(9, 6))
    else:
        fig.clf()
        plt.figure(fig.number)
    
    for model_name, history in train_histories.items():
        train_losses = history.get("train_losses", [])
        if train_losses:
            epochs = range(1, len(train_losses) + 1)
            # Get architecture from CPN_ARCHITECTURES dictionary
            n_kohonen = CPN_ARCHITECTURES.get(model_name, 0)
            plt.plot(epochs, train_losses, label=f"{model_name} (K={n_kohonen})")

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show(block=False)
    plt.pause(0.001)
    return fig


def plot_kohonen_neurons(model, x_range=(-1.5, 1.5), y_range=(-1.5, 1.5)):
    """
    Visualize the Kohonen neuron weights in 2D input space.
    
    Parameters:
    -----------
    model : CPN
        Trained CPN model
    x_range, y_range : tuple
        Range of x and y coordinates to plot
    """
    if model.input_size != 2:
        print("Visualization only available for 2D input")
        return
    
    fig = plt.figure(figsize=(8, 8))
    
    # Get Kohonen weights
    weights = model.kohonen_weights.detach().cpu().numpy()
    
    # Plot neuron positions
    plt.scatter(weights[:, 0], weights[:, 1], c='red', s=100, 
                marker='x', linewidths=2, label='Kohonen Neurons')
    
    # Add neuron indices
    for i, (wx, wy) in enumerate(weights):
        plt.annotate(f'{i}', (wx, wy), xytext=(5, 5), 
                    textcoords='offset points', fontsize=8)
    
    plt.xlim(x_range)
    plt.ylim(y_range)
    plt.xlabel('Input Dimension 1')
    plt.ylabel('Input Dimension 2')
    plt.title(f'Kohonen Neuron Positions (n={model.n_kohonen})')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show(block=False)
    return fig


def get_neuron_statistics(model):
    """
    Get statistics about Kohonen neuron usage.
    
    Returns:
    --------
    stats : dict
        Dictionary with neuron activation statistics
    """
    counts = model.neuron_counts.cpu().numpy()
    total = counts.sum()
    
    if total == 0:
        return {
            "total_activations": 0,
            "active_neurons": 0,
            "inactive_neurons": model.n_kohonen,
            "utilization_rate": 0.0,
            "max_activations": 0,
            "min_activations": 0,
            "mean_activations": 0.0,
        }
    
    return {
        "total_activations": int(total),
        "active_neurons": int((counts > 0).sum()),
        "inactive_neurons": int((counts == 0).sum()),
        "utilization_rate": float((counts > 0).sum() / model.n_kohonen),
        "max_activations": int(counts.max()),
        "min_activations": int(counts[counts > 0].min()) if (counts > 0).any() else 0,
        "mean_activations": float(counts[counts > 0].mean()) if (counts > 0).any() else 0.0,
    }


__all__ = [
    "CPN_ARCHITECTURES",
    "CPN",
    "prepare_data",
    "train_model",
    "predict",
    "evaluate_model",
    "confusion_counts",
    "plot_learning_curves",
    "plot_kohonen_neurons",
    "get_neuron_statistics",
]
