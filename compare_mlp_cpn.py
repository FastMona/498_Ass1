#!/usr/bin/env python3
"""
Comparison script for MLP and CPN architectures.
Allows training and evaluation of both neural network types on the same dataset.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

from data import generate_intertwined_spirals, DATA_PARAMS
from MLPx6 import MLP, MLP_ARCHITECTURES, prepare_data as mlp_prepare_data, train_model as mlp_train_model, evaluate_model as mlp_evaluate_model, confusion_counts as mlp_confusion_counts
from CPNx6 import CPN, CPN_ARCHITECTURES, prepare_data as cpn_prepare_data, train_model as cpn_train_model, evaluate_model as cpn_evaluate_model, confusion_counts as cpn_confusion_counts


def compare_single_models(spirals_data, mlp_arch_name="MLP_2", cpn_arch_name="CPN_2", 
                          epochs=100, batch_size=32, verbose=True):
    """
    Train and compare a single MLP and CPN architecture.
    
    Parameters:
    -----------
    spirals_data : list
        Dataset from generate_intertwined_spirals()
    mlp_arch_name : str
        Name of MLP architecture from MLP_ARCHITECTURES
    cpn_arch_name : str
        Name of CPN architecture from CPN_ARCHITECTURES
    epochs : int
        Number of training epochs
    batch_size : int
        Batch size for training
    verbose : bool
        Print detailed progress
    
    Returns:
    --------
    results : dict
        Comparison results including accuracy, training time, confusion matrix
    """
    print(f"\n{'='*70}")
    print(f"COMPARING {mlp_arch_name} vs {cpn_arch_name}")
    print(f"{'='*70}")
    
    # Prepare data (using same splits for fair comparison)
    np.random.seed(42)
    torch.manual_seed(42)
    
    train_loader, val_loader, test_loader = mlp_prepare_data(
        spirals_data, test_split=0.2, val_split=0.2, batch_size=batch_size
    )
    
    results = {}
    
    # ===== Train MLP =====
    print(f"\n--- Training {mlp_arch_name} {MLP_ARCHITECTURES[mlp_arch_name]} ---")
    mlp = MLP(hidden_layers=MLP_ARCHITECTURES[mlp_arch_name], input_size=2, output_size=1)
    
    mlp, mlp_train_losses, mlp_val_losses, mlp_test_losses, mlp_train_time_ms = mlp_train_model(
        mlp, train_loader, val_loader, test_loader, 
        epochs=epochs, lr=0.001, patience=10, verbose=verbose
    )
    
    mlp_accuracy = mlp_evaluate_model(mlp, test_loader)
    mlp_tn, mlp_fp, mlp_fn, mlp_tp = mlp_confusion_counts(mlp, test_loader)
    
    results['mlp'] = {
        'model': mlp,
        'architecture': MLP_ARCHITECTURES[mlp_arch_name],
        'train_losses': mlp_train_losses,
        'val_losses': mlp_val_losses,
        'test_losses': mlp_test_losses,
        'train_time_ms': mlp_train_time_ms,
        'train_time_s': mlp_train_time_ms / 1000.0,
        'accuracy': mlp_accuracy,
        'confusion': {'TN': mlp_tn, 'FP': mlp_fp, 'FN': mlp_fn, 'TP': mlp_tp},
        'epochs_trained': len(mlp_train_losses)
    }
    
    print(f"MLP Accuracy: {mlp_accuracy:.4f}")
    print(f"MLP Training Time: {results['mlp']['train_time_s']:.2f}s")
    
    # ===== Train CPN =====
    print(f"\n--- Training {cpn_arch_name} (K={CPN_ARCHITECTURES[cpn_arch_name]}) ---")
    cpn = CPN(n_kohonen=CPN_ARCHITECTURES[cpn_arch_name], input_size=2, output_size=1)
    
    cpn, cpn_train_losses, cpn_val_losses, cpn_test_losses, cpn_train_time_ms = cpn_train_model(
        cpn, train_loader, val_loader, test_loader,
        epochs=epochs, lr_grossberg=0.001, patience=10, verbose=verbose
    )
    
    cpn_accuracy = cpn_evaluate_model(cpn, test_loader)
    cpn_tn, cpn_fp, cpn_fn, cpn_tp = cpn_confusion_counts(cpn, test_loader)
    
    results['cpn'] = {
        'model': cpn,
        'architecture': CPN_ARCHITECTURES[cpn_arch_name],
        'train_losses': cpn_train_losses,
        'val_losses': cpn_val_losses,
        'test_losses': cpn_test_losses,
        'train_time_ms': cpn_train_time_ms,
        'train_time_s': cpn_train_time_ms / 1000.0,
        'accuracy': cpn_accuracy,
        'confusion': {'TN': cpn_tn, 'FP': cpn_fp, 'FN': cpn_fn, 'TP': cpn_tp},
        'epochs_trained': len(cpn_train_losses)
    }
    
    print(f"CPN Accuracy: {cpn_accuracy:.4f}")
    print(f"CPN Training Time: {results['cpn']['train_time_s']:.2f}s")
    
    # ===== Summary Comparison =====
    print(f"\n{'='*70}")
    print("COMPARISON SUMMARY")
    print(f"{'='*70}")
    
    print(f"\n{'Metric':<25} {'MLP':<20} {'CPN':<20} {'Winner':<10}")
    print("-" * 75)
    
    acc_winner = "MLP" if mlp_accuracy > cpn_accuracy else "CPN" if cpn_accuracy > mlp_accuracy else "Tie"
    print(f"{'Accuracy':<25} {mlp_accuracy:<20.4f} {cpn_accuracy:<20.4f} {acc_winner:<10}")
    
    time_winner = "MLP" if results['mlp']['train_time_s'] < results['cpn']['train_time_s'] else "CPN"
    print(f"{'Training Time (s)':<25} {results['mlp']['train_time_s']:<20.2f} {results['cpn']['train_time_s']:<20.2f} {time_winner:<10}")
    
    epochs_winner = "MLP" if results['mlp']['epochs_trained'] < results['cpn']['epochs_trained'] else "CPN" if results['cpn']['epochs_trained'] < results['mlp']['epochs_trained'] else "Tie"
    print(f"{'Epochs to Converge':<25} {results['mlp']['epochs_trained']:<20} {results['cpn']['epochs_trained']:<20} {epochs_winner:<10}")
    
    mlp_final_loss = mlp_test_losses[-1]
    cpn_final_loss = cpn_test_losses[-1]
    loss_winner = "MLP" if mlp_final_loss < cpn_final_loss else "CPN"
    print(f"{'Final Test Loss':<25} {mlp_final_loss:<20.4f} {cpn_final_loss:<20.4f} {loss_winner:<10}")
    
    print("\nConfusion Matrix Comparison:")
    print(f"{'Model':<10} {'TN':<8} {'FP':<8} {'FN':<8} {'TP':<8} {'Precision':<12} {'Recall':<12}")
    print("-" * 75)
    
    mlp_precision = mlp_tp / (mlp_tp + mlp_fp) if (mlp_tp + mlp_fp) > 0 else 0
    mlp_recall = mlp_tp / (mlp_tp + mlp_fn) if (mlp_tp + mlp_fn) > 0 else 0
    print(f"{'MLP':<10} {mlp_tn:<8} {mlp_fp:<8} {mlp_fn:<8} {mlp_tp:<8} {mlp_precision:<12.4f} {mlp_recall:<12.4f}")
    
    cpn_precision = cpn_tp / (cpn_tp + cpn_fp) if (cpn_tp + cpn_fp) > 0 else 0
    cpn_recall = cpn_tp / (cpn_tp + cpn_fn) if (cpn_tp + cpn_fn) > 0 else 0
    print(f"{'CPN':<10} {cpn_tn:<8} {cpn_fp:<8} {cpn_fn:<8} {cpn_tp:<8} {cpn_precision:<12.4f} {cpn_recall:<12.4f}")
    
    return results


def plot_comparison(results, title="MLP vs CPN Learning Curves"):
    """
    Plot training curves comparing MLP and CPN.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Training loss
    ax1 = axes[0]
    mlp_epochs = range(1, len(results['mlp']['train_losses']) + 1)
    cpn_epochs = range(1, len(results['cpn']['train_losses']) + 1)
    
    ax1.plot(mlp_epochs, results['mlp']['train_losses'], 'b-', label='MLP Train', linewidth=2)
    ax1.plot(mlp_epochs, results['mlp']['val_losses'], 'b--', label='MLP Val', linewidth=1.5)
    ax1.plot(cpn_epochs, results['cpn']['train_losses'], 'r-', label='CPN Train', linewidth=2)
    ax1.plot(cpn_epochs, results['cpn']['val_losses'], 'r--', label='CPN Val', linewidth=1.5)
    
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training & Validation Loss')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Test loss
    ax2 = axes[1]
    ax2.plot(mlp_epochs, results['mlp']['test_losses'], 'b-', label='MLP Test', linewidth=2)
    ax2.plot(cpn_epochs, results['cpn']['test_losses'], 'r-', label='CPN Test', linewidth=2)
    
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.set_title('Test Loss')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()
    
    return fig


def compare_all_architectures(spirals_data, epochs=100, batch_size=32):
    """
    Compare all MLP architectures with all CPN architectures.
    Creates a comprehensive comparison table.
    """
    print(f"\n{'='*80}")
    print("COMPREHENSIVE ARCHITECTURE COMPARISON")
    print(f"{'='*80}")
    
    # Prepare data once
    np.random.seed(42)
    torch.manual_seed(42)
    train_loader, val_loader, test_loader = mlp_prepare_data(
        spirals_data, test_split=0.2, val_split=0.2, batch_size=batch_size
    )
    
    mlp_results = {}
    cpn_results = {}
    
    # Train all MLPs
    print("\n--- Training MLP Architectures ---")
    for mlp_name, mlp_arch in MLP_ARCHITECTURES.items():
        print(f"\nTraining {mlp_name} {mlp_arch}...")
        mlp = MLP(hidden_layers=mlp_arch, input_size=2, output_size=1)
        
        mlp, train_losses, val_losses, test_losses, train_time_ms = mlp_train_model(
            mlp, train_loader, val_loader, test_loader,
            epochs=epochs, lr=0.001, patience=10, verbose=False
        )
        
        accuracy = mlp_evaluate_model(mlp, test_loader)
        
        mlp_results[mlp_name] = {
            'architecture': mlp_arch,
            'accuracy': accuracy,
            'train_time_s': train_time_ms / 1000.0,
            'epochs': len(train_losses),
            'final_loss': test_losses[-1] if test_losses else float('inf')
        }
        print(f"  Accuracy: {accuracy:.4f}, Time: {mlp_results[mlp_name]['train_time_s']:.2f}s")
    
    # Train all CPNs
    print("\n--- Training CPN Architectures ---")
    for cpn_name, cpn_kohonen in CPN_ARCHITECTURES.items():
        print(f"\nTraining {cpn_name} (K={cpn_kohonen})...")
        cpn = CPN(n_kohonen=cpn_kohonen, input_size=2, output_size=1)
        
        cpn, train_losses, val_losses, test_losses, train_time_ms = cpn_train_model(
            cpn, train_loader, val_loader, test_loader,
            epochs=epochs, lr_grossberg=0.001, patience=10, verbose=False
        )
        
        accuracy = cpn_evaluate_model(cpn, test_loader)
        
        cpn_results[cpn_name] = {
            'architecture': cpn_kohonen,
            'accuracy': accuracy,
            'train_time_s': train_time_ms / 1000.0,
            'epochs': len(train_losses),
            'final_loss': test_losses[-1] if test_losses else float('inf')
        }
        print(f"  Accuracy: {accuracy:.4f}, Time: {cpn_results[cpn_name]['train_time_s']:.2f}s")
    
    # Print comparison table
    print(f"\n{'='*80}")
    print("MLP RESULTS")
    print(f"{'='*80}")
    print(f"{'Model':<12} {'Architecture':<20} {'Accuracy':<12} {'Time (s)':<12} {'Epochs':<8} {'Final Loss':<12}")
    print("-" * 80)
    for name, res in mlp_results.items():
        print(f"{name:<12} {str(res['architecture']):<20} {res['accuracy']:<12.4f} {res['train_time_s']:<12.2f} {res['epochs']:<8} {res['final_loss']:<12.4f}")
    
    print(f"\n{'='*80}")
    print("CPN RESULTS")
    print(f"{'='*80}")
    print(f"{'Model':<12} {'Kohonen':<20} {'Accuracy':<12} {'Time (s)':<12} {'Epochs':<8} {'Final Loss':<12}")
    print("-" * 80)
    for name, res in cpn_results.items():
        print(f"{name:<12} {str(res['architecture']):<20} {res['accuracy']:<12.4f} {res['train_time_s']:<12.2f} {res['epochs']:<8} {res['final_loss']:<12.4f}")
    
    # Find best models
    best_mlp = max(mlp_results.items(), key=lambda x: x[1]['accuracy'])
    best_cpn = max(cpn_results.items(), key=lambda x: x[1]['accuracy'])
    
    print(f"\n{'='*80}")
    print("BEST MODELS")
    print(f"{'='*80}")
    print(f"Best MLP: {best_mlp[0]} with accuracy {best_mlp[1]['accuracy']:.4f}")
    print(f"Best CPN: {best_cpn[0]} with accuracy {best_cpn[1]['accuracy']:.4f}")
    
    if best_mlp[1]['accuracy'] > best_cpn[1]['accuracy']:
        print(f"\nOverall Winner: MLP ({best_mlp[0]}) by {(best_mlp[1]['accuracy'] - best_cpn[1]['accuracy'])*100:.2f}%")
    elif best_cpn[1]['accuracy'] > best_mlp[1]['accuracy']:
        print(f"\nOverall Winner: CPN ({best_cpn[0]}) by {(best_cpn[1]['accuracy'] - best_mlp[1]['accuracy'])*100:.2f}%")
    else:
        print("\nResult: Tie!")
    
    return mlp_results, cpn_results


def plot_architecture_comparison(mlp_results, cpn_results):
    """
    Plot comparison of all architectures.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Extract data
    mlp_names = list(mlp_results.keys())
    mlp_accuracies = [mlp_results[n]['accuracy'] for n in mlp_names]
    mlp_times = [mlp_results[n]['train_time_s'] for n in mlp_names]
    
    cpn_names = list(cpn_results.keys())
    cpn_accuracies = [cpn_results[n]['accuracy'] for n in cpn_names]
    cpn_times = [cpn_results[n]['train_time_s'] for n in cpn_names]
    
    # Accuracy comparison
    ax1 = axes[0, 0]
    x_mlp = np.arange(len(mlp_names))
    x_cpn = np.arange(len(cpn_names))
    ax1.bar(x_mlp - 0.2, mlp_accuracies, 0.4, label='MLP', color='blue', alpha=0.7)
    ax1.bar(x_cpn + 0.2, cpn_accuracies, 0.4, label='CPN', color='red', alpha=0.7)
    ax1.set_xlabel('Architecture Index')
    ax1.set_ylabel('Accuracy')
    ax1.set_title('Accuracy Comparison')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Training time comparison
    ax2 = axes[0, 1]
    ax2.bar(x_mlp - 0.2, mlp_times, 0.4, label='MLP', color='blue', alpha=0.7)
    ax2.bar(x_cpn + 0.2, cpn_times, 0.4, label='CPN', color='red', alpha=0.7)
    ax2.set_xlabel('Architecture Index')
    ax2.set_ylabel('Training Time (s)')
    ax2.set_title('Training Time Comparison')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Accuracy vs Time scatter
    ax3 = axes[1, 0]
    ax3.scatter(mlp_times, mlp_accuracies, s=100, c='blue', alpha=0.7, label='MLP', marker='o')
    ax3.scatter(cpn_times, cpn_accuracies, s=100, c='red', alpha=0.7, label='CPN', marker='s')
    for i, name in enumerate(mlp_names):
        ax3.annotate(name, (mlp_times[i], mlp_accuracies[i]), fontsize=8, alpha=0.7)
    for i, name in enumerate(cpn_names):
        ax3.annotate(name, (cpn_times[i], cpn_accuracies[i]), fontsize=8, alpha=0.7)
    ax3.set_xlabel('Training Time (s)')
    ax3.set_ylabel('Accuracy')
    ax3.set_title('Efficiency: Accuracy vs Training Time')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Loss comparison
    ax4 = axes[1, 1]
    mlp_losses = [mlp_results[n]['final_loss'] for n in mlp_names]
    cpn_losses = [cpn_results[n]['final_loss'] for n in cpn_names]
    ax4.bar(x_mlp - 0.2, mlp_losses, 0.4, label='MLP', color='blue', alpha=0.7)
    ax4.bar(x_cpn + 0.2, cpn_losses, 0.4, label='CPN', color='red', alpha=0.7)
    ax4.set_xlabel('Architecture Index')
    ax4.set_ylabel('Final Test Loss')
    ax4.set_title('Final Loss Comparison')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.suptitle('MLP vs CPN: Comprehensive Architecture Comparison', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()
    
    return fig


def main():
    """
    Interactive comparison tool for MLP vs CPN architectures.
    """
    print("="*80)
    print("MLP vs CPN Architecture Comparison Tool")
    print("="*80)
    
    # Generate dataset
    print("\nGenerating spiral dataset...")
    spirals_data = generate_intertwined_spirals(
        n=DATA_PARAMS['n'],
        noise_std=DATA_PARAMS['noise_std'],
        seed=DATA_PARAMS['seed'],
        plot=False
    )
    print(f"Dataset size: {len(spirals_data)} points")
    
    while True:
        print("\n" + "="*80)
        print("Options:")
        print("  1. Compare single MLP vs CPN")
        print("  2. Compare all architectures")
        print("  3. Custom comparison")
        print("  4. Exit")
        print("="*80)
        
        choice = input("\nSelect option (1-4): ").strip()
        
        if choice == "1":
            # Default comparison
            results = compare_single_models(spirals_data, "MLP_2", "CPN_2", epochs=100, verbose=True)
            plot_comparison(results)
            
        elif choice == "2":
            # Compare all architectures
            mlp_results, cpn_results = compare_all_architectures(spirals_data, epochs=100)
            plot_architecture_comparison(mlp_results, cpn_results)
            
        elif choice == "3":
            # Custom comparison
            print("\nAvailable MLP architectures:")
            for i, (name, arch) in enumerate(MLP_ARCHITECTURES.items(), 1):
                print(f"  {i}. {name}: {arch}")
            mlp_choice = input("Select MLP (1-7): ").strip()
            mlp_name = list(MLP_ARCHITECTURES.keys())[int(mlp_choice) - 1]
            
            print("\nAvailable CPN architectures:")
            for i, (name, kohonen) in enumerate(CPN_ARCHITECTURES.items(), 1):
                print(f"  {i}. {name}: {kohonen} Kohonen neurons")
            cpn_choice = input("Select CPN (1-7): ").strip()
            cpn_name = list(CPN_ARCHITECTURES.keys())[int(cpn_choice) - 1]
            
            epochs = int(input("Number of epochs (default 100): ").strip() or "100")
            
            results = compare_single_models(spirals_data, mlp_name, cpn_name, epochs=epochs, verbose=True)
            plot_comparison(results)
            
        elif choice == "4":
            print("\nExiting...")
            break
        else:
            print("\nInvalid choice. Please try again.")


if __name__ == "__main__":
    main()
