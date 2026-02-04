import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt


# ============================
# Data Generation Function
# ============================
def generate_intertwined_spirals(n=600, turns=1.25, a=0.32, noise_std=0.08, plot=False):
    """
    Generate two intertwined spirals.
    
    Parameters:
    -----------
    n : int
        Number of points per spiral (default 600)
    turns : float
        Number of spiral turns (default 1.25)
    a : float
        Radial scaling factor (default 0.32)
    noise_std : float
        Standard deviation of Gaussian noise (default 0.08)
    plot : bool
        If True, plot the spirals (default False)
    
    Returns:
    --------
    list of tuples
        1200 tuples total: 600 for C1 (labeled 0) and 600 for C2 (labeled 1)
        Each tuple: (x, y, label)
    """
    
    # Parameter along spiral
    theta = np.linspace(0, -2*np.pi*turns, n)
    
    # Radius as function of angle
    r = a * theta
    
    # Spiral C1
    x1_c1 = r * np.cos(theta)
    y1_c1 = r * np.sin(theta)
    
    # Spiral C2 (phase-shifted by pi)
    x1_c2 = r * np.cos(theta + np.pi)
    y1_c2 = r * np.sin(theta + np.pi)
    
    # Add noise
    x1_c1 += noise_std * np.random.randn(n)
    y1_c1 += noise_std * np.random.randn(n)
    x1_c2 += noise_std * np.random.randn(n)
    y1_c2 += noise_std * np.random.randn(n)
    
    # Center whole figure at (0, -0.5)
    y1_c1 -= 0.5
    y1_c2 -= 0.5
    
    # Offset spirals horizontally so inner tips don't touch border at x=0
    x1_c1 -= 0.15  # shift C1 left
    x1_c2 += 0.15  # shift C2 right
    
    # Create tuples: (x, y, label)
    c1_tuples = [(x1_c1[i], y1_c1[i], 0) for i in range(n)]
    c2_tuples = [(x1_c2[i], y1_c2[i], 1) for i in range(n)]
    
    # Combine and return
    data = c1_tuples + c2_tuples
    
    # Optional: plot
    if plot:
        plt.figure(figsize=(8, 8))
        plt.plot(x1_c1, y1_c1, '.', label='C1')
        plt.plot(x1_c2, y1_c2, '.', label='C2')
        plt.xlabel('$x_1$')
        plt.ylabel('$x_2$')
        plt.title('Two Intertwined Spirals')
        plt.axis('equal')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.show()
    
    return data


# ============================
# Data Generation Configuration (easily modifiable)
# ============================
DATA_PARAMS = {
    "n": 10000,         # Number of points per spiral
    "turns": 1.25,         # Number of spiral turns
    "a": 0.30,            # Radial scaling factor
    "noise_std": 0.1, # Standard deviation of Gaussian noise
}

# ============================
# MLP Architectures (easily modifiable)
# ============================
MLP_ARCHITECTURES = {
    "MLP_1": [128, 128, 64],              # 3 hidden layers
    "MLP_2": [128, 64, 64],               # 3 hidden layers
    "MLP_3": [64, 64, 64, 64, 64],        # 5 hidden layers
}


class MLP(nn.Module):
    """
    Flexible Multi-Layer Perceptron that supports 1-5 hidden layers.
    
    Parameters:
    -----------
    hidden_layers : list
        List of hidden layer sizes (e.g., [128, 128, 64])
        Length can be 1 to 5
    input_size : int
        Input dimension (default 2 for 2D coordinates)
    output_size : int
        Output dimension (default 1 for binary classification)
    """
    def __init__(self, hidden_layers, input_size=2, output_size=1):
        super(MLP, self).__init__()
        
        # Validate hidden layers
        if not isinstance(hidden_layers, list) or len(hidden_layers) == 0:
            raise ValueError("hidden_layers must be a non-empty list")
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
            patience_counter = 0
        else:
            patience_counter += 1
        
        if verbose and (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}")
        
        # Stop if patience exceeded
        if patience_counter >= patience:
            if verbose:
                print(f"Early stopping at epoch {epoch+1} (patience={patience} exceeded)")
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


if __name__ == "__main__":
    print("="*60)
    print("INTERTWINED SPIRALS CLASSIFICATION WITH MLPs")
    print("="*60)
    
    # Generate spiral data
    print("\nGenerating spiral data...")
    spirals = generate_intertwined_spirals(**DATA_PARAMS)
    total_points = len(spirals)
    points_per_spiral = DATA_PARAMS["n"]
    print(f"Total points: {total_points} ({points_per_spiral} per spiral)")
    print(f"Data parameters: {DATA_PARAMS}")
    
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
    plt.ion()
    
    while True:
        print("\n" + "="*60)
        print("MAIN MENU")
        print("="*60)
        print("1. Train MLPs")
        print("2. Batch Test (evaluate on test set)")
        print("3. Single Point Test (classify one point)")
        print("4. Display Data Plot")
        print("5. Exit")
        print("="*60)
        
        choice = input("Select option (1-5): ").strip()
        
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
            for model_name in models_to_test:
                model = trained_models[model_name]
                accuracy = evaluate_model(model, test_loader)
                print(f"{model_name:20s} - Test Accuracy: {accuracy*100:.2f}%")
        
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
            if plot_fig is None or not plt.fignum_exists(plot_fig.number):
                plot_fig = plt.figure(figsize=(8, 8))
            else:
                plot_fig.clf()
                plt.figure(plot_fig.number)
            plt.plot(x_data[c1_mask, 0], x_data[c1_mask, 1], '.', label='C1', alpha=0.6)
            plt.plot(x_data[c2_mask, 0], x_data[c2_mask, 1], '.', label='C2', alpha=0.6)
            plt.xlabel('$x_1$')
            plt.ylabel('$x_2$')
            plt.title('Two Intertwined Spirals Dataset')
            plt.axis('equal')
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.tight_layout()
            plt.show(block=False)
            plt.pause(0.001)
        
        elif choice == "5":
            print("\nExiting. Goodbye!")
            break
        
        else:
            print("Invalid option. Please select 1-5.")
