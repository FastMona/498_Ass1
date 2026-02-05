import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime


# ============================
# Region Definitions
# ============================
def _c1_predicate(x, y, r_inner, r_outer, shift):
    """C1: Left half-annulus + right half disk centered at (0, -shift)."""
    r2_origin = x ** 2 + y ** 2
    r2_shifted = x ** 2 + (y + shift) ** 2
    in_left_annulus = (x <= 0) & (r2_origin >= r_inner ** 2) & (r2_origin <= r_outer ** 2)
    in_right_inner_disk = (x >= 0) & (r2_shifted <= r_inner ** 2)
    return in_left_annulus | in_right_inner_disk


def _c2_predicate(x, y, r_inner, r_outer, shift):
    """C2: Right half-annulus centered at (0, -shift) + left half disk at origin."""
    r2_origin = x ** 2 + y ** 2
    r2_shifted = x ** 2 + (y + shift) ** 2
    in_right_annulus = (x >= 0) & (r2_shifted >= r_inner ** 2) & (r2_shifted <= r_outer ** 2)
    in_left_inner_disk = (x <= 0) & (r2_origin <= r_inner ** 2)
    return in_right_annulus | in_left_inner_disk


# ============================
# Data Generation Function
# ============================
def _plot_interlocked_region_boundaries(
    ax,
    c1_predicate,
    c2_predicate,
    bounds,
    grid_res=400,
    remove_x0_ranges=None
):
    xmin, xmax, ymin, ymax = bounds
    xs = np.linspace(xmin, xmax, grid_res)
    ys = np.linspace(ymin, ymax, grid_res)
    xx, yy = np.meshgrid(xs, ys)

    c1_mask = c1_predicate(xx, yy).astype(float)
    c2_mask = c2_predicate(xx, yy).astype(float)

    # Create combined region map for shading (before masking)
    region_map = np.zeros_like(xx)
    region_map[c1_mask > 0.5] = 1  # C1
    region_map[c2_mask > 0.5] = 2  # C2

    # Shade regions with light colors
    from matplotlib.colors import ListedColormap
    colors_list = ['white', '#CCE5FF', '#FFE5CC']  # C3 (white), C1 (light blue), C2 (light orange)
    cmap = ListedColormap(colors_list)
    ax.contourf(xx, yy, region_map, levels=[0, 0.5, 1.5, 2.5], cmap=cmap, alpha=0.7)

    # Apply masking only for boundary lines
    if remove_x0_ranges:
        dx = xs[1] - xs[0]
        near_x0 = np.abs(xx) <= (1.5 * dx)
        for y_min, y_max in remove_x0_ranges:
            in_range = (yy >= y_min) & (yy <= y_max)
            c1_mask[near_x0 & in_range] = np.nan
            c2_mask[near_x0 & in_range] = np.nan

    ax.contour(xx, yy, c1_mask, levels=[0.5], colors='k', linewidths=3)
    ax.contour(xx, yy, c2_mask, levels=[0.5], colors='k', linewidths=3)


def _plot_both_sampling_methods(rnd_data, ctr_data, r_inner, r_outer, shift):
    """Plot side-by-side comparison of RND and CTR sampling methods."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Extract data for RND
    x_c1_rnd = [p[0] for p in rnd_data if p[2] == 0]
    y_c1_rnd = [p[1] for p in rnd_data if p[2] == 0]
    x_c2_rnd = [p[0] for p in rnd_data if p[2] == 1]
    y_c2_rnd = [p[1] for p in rnd_data if p[2] == 1]
    
    # Extract data for CTR
    x_c1_ctr = [p[0] for p in ctr_data if p[2] == 0]
    y_c1_ctr = [p[1] for p in ctr_data if p[2] == 0]
    x_c2_ctr = [p[0] for p in ctr_data if p[2] == 1]
    y_c2_ctr = [p[1] for p in ctr_data if p[2] == 1]
    
    # Create predicates for boundaries
    def c1_pred(x, y):
        return _c1_predicate(x, y, r_inner, r_outer, shift)
    def c2_pred(x, y):
        return _c2_predicate(x, y, r_inner, r_outer, shift)
    
    bounds = (-r_outer - 0.5, r_outer + 0.5, -shift - r_outer - 0.5, r_outer + 0.5)
    
    # Plot RND (left subplot)
    ax1.plot(x_c1_rnd, y_c1_rnd, '.', label='C1', alpha=0.6, markersize=3)
    ax1.plot(x_c2_rnd, y_c2_rnd, '.', label='C2', alpha=0.6, markersize=3)
    _plot_interlocked_region_boundaries(
        ax1, c1_pred, c2_pred, bounds, remove_x0_ranges=[(0, r_inner)]
    )
    ax1.set_xlabel('$x_1$')
    ax1.set_ylabel('$x_2$')
    ax1.set_title('RND (Uniform Random Sampling)')
    ax1.set_aspect('equal', 'box')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Plot CTR (right subplot)
    ax2.plot(x_c1_ctr, y_c1_ctr, '.', label='C1', alpha=0.6, markersize=3)
    ax2.plot(x_c2_ctr, y_c2_ctr, '.', label='C2', alpha=0.6, markersize=3)
    _plot_interlocked_region_boundaries(
        ax2, c1_pred, c2_pred, bounds, remove_x0_ranges=[(0, r_inner)]
    )
    ax2.set_xlabel('$x_1$')
    ax2.set_ylabel('$x_2$')
    ax2.set_title('CTR (Center-Weighted Sampling at (0, -0.5))')
    ax2.set_aspect('equal', 'box')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    plt.show()


def generate_intertwined_spirals(
    n=600,
    r_inner=1.0,
    r_outer=2.0,
    shift=1.0,
    noise_std=0.0,
    seed=None,
    include_inner_caps=False,
    plot=False,
    sampling_method="RND"
):
    """
    Generate two interlocked regions that form symmetric spiraling shapes.

    Region C1 (blue): Left half-annulus [r_inner, r_outer] centered at origin
                      + right half disk [0, r_inner] centered at (0, -shift)
    Region C2 (orange): Right half-annulus [r_inner, r_outer] centered at (0, -shift)
                        + left half disk [0, r_inner] centered at origin

    The regions interlock symmetrically around the point (0, -shift/2).

    Parameters:
    -----------
    n : int
        Number of points per region (default 600)
    r_inner : float
        Inner radius of annuli (default 1.0)
    r_outer : float
        Outer radius of annuli (default 2.0)
    shift : float
        Downward shift of right half regions (default 1.0)
    noise_std : float
        Standard deviation of Gaussian noise (default 0.0)
    seed : int or None
        Random seed for reproducibility (default None)
    include_inner_caps : bool
        Unused legacy parameter (default False)
    plot : bool
        If True, plot the regions with boundaries (default False)
    sampling_method : str
        "RND" for uniform random sampling
        "CTR" for center-weighted (shotgun) sampling around (0, -0.5)
        "BOTH" to generate both and show side-by-side plots

    Returns:
    --------
    list of tuples or tuple of two lists
        If sampling_method is "RND" or "CTR": list of 2n tuples (x, y, label)
        If sampling_method is "BOTH": tuple of (rnd_data, ctr_data)
    """
    rng = np.random.default_rng(seed)

    # Use predicate helper functions with partial application
    def c1_predicate(x, y):
        return _c1_predicate(x, y, r_inner, r_outer, shift)

    def c2_predicate(x, y):
        return _c2_predicate(x, y, r_inner, r_outer, shift)

    def _sample_half_annulus(n_samples, theta_min, theta_max, y_shift=0):
        """Sample uniformly from a half-annulus in polar coordinates."""
        theta = rng.uniform(theta_min, theta_max, n_samples)
        r = np.sqrt(rng.uniform(r_inner ** 2, r_outer ** 2, n_samples))
        return r * np.cos(theta), r * np.sin(theta) + y_shift

    def _sample_half_disk(n_samples, theta_min, theta_max, y_shift=0):
        """Sample uniformly from a half-disk in polar coordinates."""
        theta = rng.uniform(theta_min, theta_max, n_samples)
        r = np.sqrt(rng.uniform(0, r_inner ** 2, n_samples))
        return r * np.cos(theta), r * np.sin(theta) + y_shift

    def _sample_c1(n_samples):
        """Sample C1: left annulus + right half disk at (0, -shift)."""
        # Proportional split by area
        area_annulus = np.pi * (r_outer ** 2 - r_inner ** 2) / 2
        area_disk = np.pi * r_inner ** 2 / 2
        n_annulus = int(n_samples * area_annulus / (area_annulus + area_disk))
        n_disk = n_samples - n_annulus
        
        x_ann, y_ann = _sample_half_annulus(n_annulus, np.pi / 2, 3 * np.pi / 2)
        x_disk, y_disk = _sample_half_disk(n_disk, -np.pi / 2, np.pi / 2, -shift)
        
        return np.concatenate([x_ann, x_disk]), np.concatenate([y_ann, y_disk])

    def _sample_c2(n_samples):
        """Sample C2: right annulus at (0, -shift) + left half disk at origin."""
        # Proportional split by area
        area_annulus = np.pi * (r_outer ** 2 - r_inner ** 2) / 2
        area_disk = np.pi * r_inner ** 2 / 2
        n_annulus = int(n_samples * area_annulus / (area_annulus + area_disk))
        n_disk = n_samples - n_annulus
        
        x_ann, y_ann = _sample_half_annulus(n_annulus, -np.pi / 2, np.pi / 2, -shift)
        x_disk, y_disk = _sample_half_disk(n_disk, np.pi / 2, 3 * np.pi / 2)
        
        return np.concatenate([x_ann, x_disk]), np.concatenate([y_ann, y_disk])

    # Center-weighted (shotgun) sampling functions
    def _sample_c1_centered(n_samples, center_x=0, center_y=-0.5, sigma=0.8):
        """Sample C1 using center-weighted Gaussian distribution with rejection sampling."""
        x_samples, y_samples = [], []
        max_attempts = n_samples * 100  # Safety limit
        attempts = 0
        
        while len(x_samples) < n_samples and attempts < max_attempts:
            # Generate candidates from 2D Gaussian centered at (center_x, center_y)
            batch_size = (n_samples - len(x_samples)) * 3
            x_cand = rng.normal(center_x, sigma, batch_size)
            y_cand = rng.normal(center_y, sigma, batch_size)
            
            # Keep only points that fall within C1 region
            mask = c1_predicate(x_cand, y_cand)
            x_samples.extend(x_cand[mask])
            y_samples.extend(y_cand[mask])
            attempts += batch_size
        
        return np.array(x_samples[:n_samples]), np.array(y_samples[:n_samples])

    def _sample_c2_centered(n_samples, center_x=0, center_y=-0.5, sigma=0.8):
        """Sample C2 using center-weighted Gaussian distribution with rejection sampling."""
        x_samples, y_samples = [], []
        max_attempts = n_samples * 100  # Safety limit
        attempts = 0
        
        while len(x_samples) < n_samples and attempts < max_attempts:
            # Generate candidates from 2D Gaussian centered at (center_x, center_y)
            batch_size = (n_samples - len(x_samples)) * 3
            x_cand = rng.normal(center_x, sigma, batch_size)
            y_cand = rng.normal(center_y, sigma, batch_size)
            
            # Keep only points that fall within C2 region
            mask = c2_predicate(x_cand, y_cand)
            x_samples.extend(x_cand[mask])
            y_samples.extend(y_cand[mask])
            attempts += batch_size
        
        return np.array(x_samples[:n_samples]), np.array(y_samples[:n_samples])

    # Choose sampling method
    if sampling_method == "BOTH":
        # Generate both datasets
        # Random sampling
        x_left_rnd, y_left_rnd = _sample_c1(n)
        x_right_rnd, y_right_rnd = _sample_c2(n)
        
        # Center-weighted sampling
        x_left_ctr, y_left_ctr = _sample_c1_centered(n)
        x_right_ctr, y_right_ctr = _sample_c2_centered(n)
        
        if noise_std > 0:
            x_left_rnd += noise_std * rng.standard_normal(n)
            y_left_rnd += noise_std * rng.standard_normal(n)
            x_right_rnd += noise_std * rng.standard_normal(n)
            y_right_rnd += noise_std * rng.standard_normal(n)
            
            x_left_ctr += noise_std * rng.standard_normal(n)
            y_left_ctr += noise_std * rng.standard_normal(n)
            x_right_ctr += noise_std * rng.standard_normal(n)
            y_right_ctr += noise_std * rng.standard_normal(n)
        
        rnd_data = [(x_left_rnd[i], y_left_rnd[i], 0) for i in range(n)] + \
                   [(x_right_rnd[i], y_right_rnd[i], 1) for i in range(n)]
        ctr_data = [(x_left_ctr[i], y_left_ctr[i], 0) for i in range(n)] + \
                   [(x_right_ctr[i], y_right_ctr[i], 1) for i in range(n)]
        
        if plot:
            _plot_both_sampling_methods(rnd_data, ctr_data, r_inner, r_outer, shift)
        
        return rnd_data, ctr_data
    
    elif sampling_method == "CTR":
        # Center-weighted sampling only
        x_left, y_left = _sample_c1_centered(n)
        x_right, y_right = _sample_c2_centered(n)
    else:  # "RND" or default
        # Uniform random sampling
        x_left, y_left = _sample_c1(n)
        x_right, y_right = _sample_c2(n)

    if noise_std > 0:
        x_left += noise_std * rng.standard_normal(n)
        y_left += noise_std * rng.standard_normal(n)
        x_right += noise_std * rng.standard_normal(n)
        y_right += noise_std * rng.standard_normal(n)

    c1_tuples = [(x_left[i], y_left[i], 0) for i in range(n)]
    c2_tuples = [(x_right[i], y_right[i], 1) for i in range(n)]
    data = c1_tuples + c2_tuples

    if plot:
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.plot(x_left, y_left, '.', label='C1', alpha=0.6)
        ax.plot(x_right, y_right, '.', label='C2', alpha=0.6)
        bounds = (-r_outer - 0.5, r_outer + 0.5, -shift - r_outer - 0.5, r_outer + 0.5)
        _plot_interlocked_region_boundaries(
            ax,
            c1_predicate,
            c2_predicate,
            bounds,
            remove_x0_ranges=[(0, r_inner)]
        )
        ax.set_xlabel('$x_1$')
        ax.set_ylabel('$x_2$')
        ax.set_title('Interlocked Annulus Regions')
        ax.set_aspect('equal', 'box')
        ax.grid(True, alpha=0.3)
        ax.legend()
        plt.show()

    return data


# ============================
# Data Generation Configuration (easily modifiable)
# ============================
DATA_PARAMS = {
    "n": 1000,            # Number of points per region
    "r_inner": 1.0,        # Inner radius
    "r_outer": 2.0,        # Outer radius
    "shift": 1.0,          # Downward shift of right half
    "noise_std": 0.0,      # Gaussian noise (0 for clean boundaries)
    "seed": 7,
    "include_inner_caps": False,  # No special fill
    "sampling_method": "BOTH",  # "RND" (uniform random), "CTR" (center-weighted), or "BOTH" (show side-by-side)
}

# ============================
# MLP Architectures (easily modifiable)
# ============================
MLP_ARCHITECTURES = {
    "MLP_0": [],                        # No hidden layers (logistic regression)
    "MLP_1": [8],                  # 16 node hidden layer
    "MLP_2": [8, 8],                  # 64 node hidden layers
    "MLP_3": [16],                # 3 hidden layers
    "MLP_4": [16, 16 ],                # 4
    "MLP_5": [16, 16, 16],                # 5
}


class MLP(nn.Module):
    """
    Flexible Multi-Layer Perceptron that supports 1-5 hidden layers.
    
    Parameters:
    -----------
    hidden_layers : list
        List of hidden layer sizes (e.g., [128, 128, 64])
        Length can be 0 to 5; [] means no hidden layers
    input_size : int
        Input dimension (default 2 for 2D coordinates)
    output_size : int
        Output dimension (default 1 for binary classification)
    """
    def __init__(self, hidden_layers, input_size=2, output_size=1):
        super(MLP, self).__init__()
        
        # Validate hidden layers
        if not isinstance(hidden_layers, list):
            raise ValueError("hidden_layers must be a list")
        if len(hidden_layers) > 5:
            raise ValueError("Maximum 5 hidden layers supported")
        
        # Build layers
        layers = []
        prev_size = input_size
        
        # Hidden layers
        for hidden_size in hidden_layers:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            prev_size = hidden_size
        
        # Output layer
        layers.append(nn.Linear(prev_size, output_size))
        layers.append(nn.Sigmoid())
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)


def prepare_data(spirals_data, test_split=0.2, batch_size=32):
    """
    Prepare data for training.
    
    Parameters:
    -----------
    spirals_data : list of tuples
        Output from generate_intertwined_spirals()
    test_split : float
        Fraction of data for testing (default 0.2)
    batch_size : int
        Batch size for DataLoader
    
    Returns:
    --------
    train_loader, test_loader : DataLoader objects
    """
    # Extract x, y, and labels
    x_data = np.array([[x, y] for x, y, _ in spirals_data], dtype=np.float32)
    y_data = np.array([label for _, _, label in spirals_data], dtype=np.float32).reshape(-1, 1)
    
    # Convert to tensors
    x_tensor = torch.from_numpy(x_data)
    y_tensor = torch.from_numpy(y_data)
    
    # Split into train and test
    n_samples = len(spirals_data)
    n_train = int(n_samples * (1 - test_split))
    
    indices = np.random.permutation(n_samples)
    train_indices = indices[:n_train]
    test_indices = indices[n_train:]
    
    train_dataset = TensorDataset(x_tensor[train_indices], y_tensor[train_indices])
    test_dataset = TensorDataset(x_tensor[test_indices], y_tensor[test_indices])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader


def train_model(model, train_loader, test_loader, epochs=100, lr=0.001, verbose=True):
    """
    Train an MLP model.
    
    Parameters:
    -----------
    model : nn.Module
        The model to train
    train_loader : DataLoader
        Training data loader
    test_loader : DataLoader
        Testing data loader
    epochs : int
        Number of training epochs
    lr : float
        Learning rate
    verbose : bool
        Print progress
    
    Returns:
    --------
    model : trained model
    train_losses : list of training losses
    test_losses : list of testing losses
    """
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    train_losses = []
    test_losses = []
    
    # Early stopping parameters
    patience = 5
    best_loss = float('inf')
    best_model_state = None
    best_epoch = 0
    patience_counter = 0
    
    for epoch in range(epochs):
        # Training
        train_loss = 0.0
        for x_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        # Testing
        test_loss = 0.0
        with torch.no_grad():
            for x_batch, y_batch in test_loader:
                outputs = model(x_batch)
                loss = criterion(outputs, y_batch)
                test_loss += loss.item()
        
        test_loss /= len(test_loader)
        test_losses.append(test_loss)
        
        # Early stopping check
        if test_loss < best_loss:
            best_loss = test_loss
            best_model_state = model.state_dict().copy()
            best_epoch = epoch
            patience_counter = 0
        else:
            patience_counter += 1
        
        if verbose and (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}")
        
        # Stop if patience exceeded
        if patience_counter >= patience:
            if verbose:
                print(f"Early stopping at epoch {epoch+1} (patience={patience} exceeded)")
            # Restore best model state
            if best_model_state is not None:
                model.load_state_dict(best_model_state)
            # Truncate loss lists to best epoch
            train_losses = train_losses[:best_epoch + 1]
            test_losses = test_losses[:best_epoch + 1]
            break
    
    return model, train_losses, test_losses


def predict(model, x, y):
    """
    Predict class for a single point.
    
    Parameters:
    -----------
    model : nn.Module
        Trained model
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
        output = model(input_tensor).item()
    
    prediction = 1 if output > 0.5 else 0
    confidence = output if prediction == 1 else (1 - output)
    
    return prediction, confidence


def evaluate_model(model, test_loader):
    """
    Evaluate model on test set and return accuracy.
    
    Parameters:
    -----------
    model : nn.Module
        Trained model
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
            outputs = model(x_batch)
            predicted = (outputs > 0.5).float()
            total += y_batch.size(0)
            correct += (predicted == y_batch).sum().item()
    
    return correct / total


def plot_learning_curves(train_histories, title="Learning Curves", fig=None):
    """
    Plot learning curves (loss vs epochs) for one or more models.

    Parameters:
    -----------
    train_histories : dict
        {model_name: {"train_losses": [...], "test_losses": [...]}}
    title : str
        Plot title
    """
    if not train_histories:
        return

    if fig is None or not plt.fignum_exists(fig.number):
        fig = plt.figure(figsize=(9, 6))
    else:
        fig.clf()
        plt.figure(fig.number)
    for model_name, history in train_histories.items():
        train_losses = history.get("train_losses", [])
        if train_losses:
            epochs = range(1, len(train_losses) + 1)
            # Get architecture from MLP_ARCHITECTURES dictionary
            arch = MLP_ARCHITECTURES.get(model_name, [])
            plt.plot(epochs, train_losses, label=f"{arch}")

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show(block=False)
    plt.pause(0.001)
    return fig


if __name__ == "__main__":
    print("="*60)
    print("INTERTWINED SPIRALS CLASSIFICATION WITH MLPs")
    print("="*60)
    
    # Generate spiral data
    print("\nGenerating interlocked region data...")
    sampling_method = DATA_PARAMS.get("sampling_method", "RND")
    
    if sampling_method == "BOTH":
        # Generate both datasets but use RND for training
        result = generate_intertwined_spirals(
            n=DATA_PARAMS["n"],
            r_inner=DATA_PARAMS["r_inner"],
            r_outer=DATA_PARAMS["r_outer"],
            shift=DATA_PARAMS["shift"],
            noise_std=DATA_PARAMS["noise_std"],
            seed=DATA_PARAMS["seed"],
            sampling_method="BOTH",
            plot=False
        )
        spirals = result[0]  # Use RND data for training
        print(f"Generated both RND and CTR datasets for comparison")
    else:
        spirals = generate_intertwined_spirals(**DATA_PARAMS)
    
    total_points = len(spirals)
    points_per_spiral = DATA_PARAMS["n"]
    print(f"Total points: {total_points} ({points_per_spiral} per spiral)")
    print(f"Data parameters: {DATA_PARAMS}")
    print(f"Sampling method: {sampling_method}")
    
    # Extract data for plotting
    x_data = np.array([[x, y] for x, y, _ in spirals])
    c1_mask = np.array([label == 0 for _, _, label in spirals])
    c2_mask = np.array([label == 1 for _, _, label in spirals])
    
    # Prepare data
    print("Preparing data (80% train, 20% test)...")
    train_loader, test_loader = prepare_data(spirals, test_split=0.2, batch_size=32)
    
    # Create models dictionary
    trained_models = {}
    train_histories = {}
    plot_fig = None
    learning_curve_fig = None
    plt.ion()
    
    while True:
        print("\n" + "="*60)
        print("MAIN MENU")
        print("="*60)
        print("1. Train MLPs")
        print("2. Batch Test (evaluate on test set)")
        print("3. Single Point Test (classify one point)")
        print("4. Display Data Plot")
        print("0. Exit")
        print("="*60)
        
        choice = input("Select option (0-4): ").strip()
        
        if choice == "1":
            print("\n" + "="*60)
            print("TRAINING MENU")
            print("="*60)
            print("Available architectures:")
            for i, (name, arch) in enumerate(MLP_ARCHITECTURES.items(), 1):
                print(f"  {i}. {name}: {arch}")
            
            train_choice = input("\nTrain which architecture(s)? (default 'all', or '1', '1,2,3'): ").strip().lower()
            
            if train_choice == "all" or train_choice == "":
                architectures_to_train = list(MLP_ARCHITECTURES.items())
            else:
                try:
                    indices = [int(x.strip()) - 1 for x in train_choice.split(",")]
                    architectures_to_train = [
                        (name, arch) for i, (name, arch) in enumerate(MLP_ARCHITECTURES.items())
                        if i in indices
                    ]
                except (ValueError, IndexError):
                    print("Invalid input. Using all architectures.")
                    architectures_to_train = list(MLP_ARCHITECTURES.items())
            
            epochs_str = input("Number of epochs (default 100): ").strip()
            epochs = int(epochs_str) if epochs_str else 100
            
            lr_str = input("Learning rate (default 0.001): ").strip()
            lr = float(lr_str) if lr_str else 0.001
            
            train_rows = []
            for arch_name, hidden_layers in architectures_to_train:
                print(f"\n{'='*60}")
                print(f"Training {arch_name} with hidden layers: {hidden_layers}")
                print(f"{'='*60}")
                
                model = MLP(hidden_layers)
                trained_model, train_losses, test_losses = train_model(
                    model, train_loader, test_loader, epochs=epochs, lr=lr, verbose=True
                )
                
                trained_models[arch_name] = trained_model
                train_histories[arch_name] = {
                    "train_losses": train_losses,
                    "test_losses": test_losses
                }
                
                final_acc = evaluate_model(trained_model, test_loader)
                print(f"Final Test Accuracy: {final_acc*100:.2f}%")
                train_rows.append((arch_name, hidden_layers, final_acc))
            
            if train_rows:
                print("\n" + "="*60)
                print("TRAINING SUMMARY (Structure & Accuracy)")
                print("="*60)
                print(f"{'Model':12s} | {'Structure':24s} | {'Accuracy':>9s}")
                print("-"*60)
                for name, structure, acc in train_rows:
                    print(f"{name:12s} | {str(structure):24s} | {acc*100:8.2f}%")

                # Plot learning curves for trained models (separate window)
                learning_curve_fig = plot_learning_curves(
                    train_histories,
                    title="Training Learning Curves",
                    fig=learning_curve_fig
                )
        
        elif choice == "2":
            if not trained_models:
                print("\nNo trained models yet. Please train models first (option 1).")
                continue
            
            print("\n" + "="*60)
            print("BATCH TEST MENU")
            print("="*60)
            print("Trained models:")
            for i, model_name in enumerate(trained_models.keys(), 1):
                print(f"  {i}. {model_name}")
            
            test_choice = input("Test which model(s)? (default 'all', or '1', '1,2'): ").strip().lower()
            
            if test_choice == "all" or test_choice == "":
                models_to_test = list(trained_models.keys())
            else:
                try:
                    indices = [int(x.strip()) - 1 for x in test_choice.split(",")]
                    models_to_test = [
                        model_name for i, model_name in enumerate(trained_models.keys())
                        if i in indices
                    ]
                except (ValueError, IndexError):
                    print("Invalid input. Testing all models.")
                    models_to_test = list(trained_models.keys())
            
            print("\n" + "-"*60)
            print("Test Results:")
            print("-"*60)
            test_rows = []
            for model_name in models_to_test:
                model = trained_models[model_name]
                accuracy = evaluate_model(model, test_loader)
                print(f"{model_name:20s} - Test Accuracy: {accuracy*100:.2f}%")
                structure = MLP_ARCHITECTURES.get(model_name, "N/A")
                test_rows.append((model_name, structure, accuracy))

            if test_rows:
                print("\n" + "-"*60)
                print("Summary (Structure & Accuracy)")
                print("-"*60)
                print(f"{'Model':12s} | {'Structure':24s} | {'Accuracy':>9s}")
                print("-"*60)
                for name, structure, acc in test_rows:
                    print(f"{name:12s} | {str(structure):24s} | {acc*100:8.2f}%")
                
                # Save to test_results.txt
                with open('test_results.txt', 'a') as f:
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    f.write(f"\n{'='*60}\n")
                    f.write(f"Test Results - {timestamp}\n")
                    f.write(f"{'='*60}\n")
                    f.write(f"{'Model':12s} | {'Structure':24s} | {'Accuracy':>9s}\n")
                    f.write(f"{'-'*60}\n")
                    for name, structure, acc in test_rows:
                        f.write(f"{name:12s} | {str(structure):24s} | {acc*100:8.2f}%\n")
                
                print(f"\nResults saved to test_results.txt")
        
        elif choice == "3":
            if not trained_models:
                print("\nNo trained models yet. Please train models first (option 1).")
                continue
            
            print("\n" + "="*60)
            print("SINGLE POINT TEST MENU")
            print("="*60)
            print("Trained models:")
            for i, model_name in enumerate(trained_models.keys(), 1):
                print(f"  {i}. {model_name}")
            
            test_choice = input("Test which model(s)? (default 'all', or '1', '1,2'): ").strip().lower()
            
            if test_choice == "all" or test_choice == "":
                models_to_test = list(trained_models.keys())
            else:
                try:
                    indices = [int(x.strip()) - 1 for x in test_choice.split(",")]
                    models_to_test = [
                        model_name for i, model_name in enumerate(trained_models.keys())
                        if i in indices
                    ]
                except (ValueError, IndexError):
                    print("Invalid input. Testing all models.")
                    models_to_test = list(trained_models.keys())
            
            try:
                x_str = input("Enter x coordinate: ").strip()
                y_str = input("Enter y coordinate: ").strip()
                x = float(x_str)
                y = float(y_str)
                
                print("\n" + "-"*60)
                print(f"Predictions for point ({x:.4f}, {y:.4f}):")
                print("-"*60)
                for model_name in models_to_test:
                    model = trained_models[model_name]
                    pred, conf = predict(model, x, y)
                    class_name = "C1" if pred == 0 else "C2"
                    print(f"{model_name:20s} -> {class_name} (confidence: {conf:.4f})")
            
            except ValueError:
                print("Invalid input. Please enter valid numbers.")
        
        elif choice == "4":
            print("\n" + "="*60)
            print("DISPLAY DATA PLOT")
            print("="*60)
            
            # Check if we have both sampling methods
            if DATA_PARAMS.get("sampling_method") == "BOTH":
                # Regenerate with plot=True to show side-by-side
                _ = generate_intertwined_spirals(**{**DATA_PARAMS, "plot": True})
            else:
                # Single plot for RND or CTR
                if plot_fig is None or not plt.fignum_exists(plot_fig.number):
                    plot_fig = plt.figure(figsize=(8, 8))
                else:
                    plot_fig.clf()
                    plt.figure(plot_fig.number)
                plt.plot(x_data[c1_mask, 0], x_data[c1_mask, 1], '.', label='C1', alpha=0.6)
                plt.plot(x_data[c2_mask, 0], x_data[c2_mask, 1], '.', label='C2', alpha=0.6)
                # Define region predicates using helper functions
                def _c1_pred(x, y):
                    return _c1_predicate(x, y, DATA_PARAMS["r_inner"], DATA_PARAMS["r_outer"], DATA_PARAMS["shift"])

                def _c2_pred(x, y):
                    return _c2_predicate(x, y, DATA_PARAMS["r_inner"], DATA_PARAMS["r_outer"], DATA_PARAMS["shift"])

                bounds = (
                    -DATA_PARAMS["r_outer"] - 0.5,
                    DATA_PARAMS["r_outer"] + 0.5,
                    -DATA_PARAMS["shift"] - DATA_PARAMS["r_outer"] - 0.5,
                    DATA_PARAMS["r_outer"] + 0.5,
                )
                _plot_interlocked_region_boundaries(
                    plt.gca(),
                    _c1_pred,
                    _c2_pred,
                    bounds,
                    remove_x0_ranges=[(0, DATA_PARAMS["r_inner"])],
                )
                method_name = DATA_PARAMS.get("sampling_method", "RND")
                plt.xlabel('$x_1$')
                plt.ylabel('$x_2$')
                plt.title(f'Interlocked Annulus Regions Dataset ({method_name})')
                plt.axis('equal')
                plt.grid(True, alpha=0.3)
                plt.legend()
                plt.tight_layout()
                plt.show(block=False)
                plt.pause(0.001)
        
        elif choice == "0":
            print("\nExiting. Goodbye!")
            break
        
        else:
            print("Invalid option. Please select 0-4.")
