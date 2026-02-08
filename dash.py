import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime

from data import DATA_PARAMS, generate_intertwined_spirals, plot_dataset, plot_three_datasets
from MLPx6 import (
    MLP,
    MLP_ARCHITECTURES,
    prepare_data,
    train_model,
    evaluate_model,
    predict,
    plot_learning_curves,
)


def _generate_datasets(data_params, active_dataset):
    sampling_method = "ALL"
    data_params["sampling_method"] = sampling_method
    dataset_storage = {}

    if sampling_method == "ALL":
        result = generate_intertwined_spirals(
            n=data_params["n"],
            r_inner=data_params["r_inner"],
            r_outer=data_params["r_outer"],
            shift=data_params["shift"],
            noise_std=data_params["noise_std"],
            seed=data_params["seed"],
            sampling_method="ALL",
            plot=False
        )
        dataset_storage["RND"] = result[0]
        dataset_storage["CTR"] = result[1]
        dataset_storage["EDGE"] = result[2]
        if active_dataset not in dataset_storage:
            active_dataset = "RND"
        spirals = dataset_storage[active_dataset]
    else:
        spirals = generate_intertwined_spirals(**data_params)
        dataset_storage[sampling_method] = spirals
        active_dataset = sampling_method

    return dataset_storage, active_dataset, spirals, sampling_method


def _init_dataloaders(dataset_storage, batch_size=32):
    train_loaders = {}
    val_loaders = {}
    test_loaders = {}
    trained_models = {}
    train_histories = {}

    for ds_name in dataset_storage.keys():
        train_loaders[ds_name], val_loaders[ds_name], test_loaders[ds_name] = prepare_data(
            dataset_storage[ds_name], test_split=0.2, val_split=0.2, batch_size=batch_size
        )
        trained_models[ds_name] = {}
        train_histories[ds_name] = {}

    return train_loaders, val_loaders, test_loaders, trained_models, train_histories


def main():
    print("="*60)
    print("INTERTWINED SPIRALS CLASSIFICATION WITH MLPs")
    print("="*60)

    # Generate spiral data
    print("\nGenerating interlocked region data...")
    active_dataset = "RND"  # Default active dataset
    dataset_storage, active_dataset, spirals, sampling_method = _generate_datasets(DATA_PARAMS, active_dataset)
    if sampling_method == "ALL":
        print("Generated RND, CTR, and EDGE datasets")
        print(f"Active dataset for training: {active_dataset}")

    total_points = len(spirals)
    points_per_spiral = DATA_PARAMS["n"]
    print(f"Total points: {total_points} ({points_per_spiral} per spiral)")
    print(f"Data parameters: {DATA_PARAMS}")
    print(f"Sampling method: {sampling_method}")

    # Create models dictionary - organized by dataset
    train_loaders, val_loaders, test_loaders, trained_models, train_histories = _init_dataloaders(
        dataset_storage, batch_size=32
    )

    plt.ion()

    while True:
        print("\n" + "="*60)
        print("MAIN MENU")
        print("="*60)
        print("1. Create Datasets")
        print("2. Display Current Dataset")
        print("3. Single Point Test (classify one point)")
        if len(dataset_storage) > 1:
            print("4. Train on ALL Datasets")
            print("5. Cross-Dataset Comparison")
        print("6. Cleanup")
        print("0. Exit")
        print("="*60)

        max_option = "6" if len(dataset_storage) > 1 else "4"
        choice = input(f"Select option (0-{max_option}): ").strip()

        if choice == "1":
            print("\n" + "="*60)
            print("CREATE DATA SETS")
            print("="*60)
            n_str = input(f"Number of points per region (default {DATA_PARAMS['n']}): ").strip()
            if n_str:
                try:
                    n_value = int(n_str)
                    if n_value > 0:
                        DATA_PARAMS["n"] = n_value
                    else:
                        print("Invalid value. Keeping existing number of points.")
                except ValueError:
                    print("Invalid value. Keeping existing number of points.")

            dataset_storage, active_dataset, spirals, sampling_method = _generate_datasets(
                DATA_PARAMS, active_dataset
            )
            if sampling_method == "ALL":
                print("Generated RND, CTR, and EDGE datasets")

            total_points = len(spirals)
            points_per_spiral = DATA_PARAMS["n"]
            print(f"Total points: {total_points} ({points_per_spiral} per spiral)")
            print(f"Active dataset for training: {active_dataset}")

            train_loaders, val_loaders, test_loaders, trained_models, train_histories = _init_dataloaders(
                dataset_storage, batch_size=32
            )

        elif choice == "2":
            print("\n" + "="*60)
            print("DISPLAY CURRENT DATA SETS")
            print("="*60)
            if not plot_three_datasets(dataset_storage, DATA_PARAMS):
                print("Missing datasets; recreate datasets first (option 1).")
        elif choice == "3":
            if not any(any(models.values()) for models in trained_models.values()):
                print("\nNo trained models yet. Please train models first (option 4).")
                continue

            print("\n" + "="*60)
            print("SINGLE POINT TEST MENU")
            print("All trained models across RND/CTR/EDGE datasets")
            print("="*60)

            try:
                x_str = input("Enter x coordinate: ").strip()
                y_str = input("Enter y coordinate: ").strip()
                x = float(x_str)
                y = float(y_str)

                print("\n" + "-"*60)
                print(f"Predictions for point ({x:.4f}, {y:.4f}):")
                print("-"*60)

                all_conf_values = []
                for ds_name in ["RND", "CTR", "EDGE"]:
                    if ds_name not in trained_models or not trained_models[ds_name]:
                        continue
                    print(f"\nDataset: {ds_name}")
                    for model_name, model in trained_models[ds_name].items():
                        pred, conf = predict(model, x, y)
                        all_conf_values.append(conf)
                        class_name = "C1" if pred == 0 else "C2"
                        print(f"{model_name:20s} -> {class_name} (confidence: {conf:.4f})")
                if all_conf_values:
                    avg_conf = sum(all_conf_values) / len(all_conf_values)
                    sorted_conf = sorted(all_conf_values)
                    mid = len(sorted_conf) // 2
                    if len(sorted_conf) % 2 == 0:
                        median_conf = (sorted_conf[mid - 1] + sorted_conf[mid]) / 2
                    else:
                        median_conf = sorted_conf[mid]
                    print("\nSummary (all datasets)")
                    print(f"Average confidence: {avg_conf:.4f}")
                    print(f"Median confidence: {median_conf:.4f}")

            except ValueError:
                print("Invalid input. Please enter valid numbers.")

        elif choice == "4" and len(dataset_storage) > 1:
            print("\n" + "="*60)
            print("TRAIN ON ALL DATASETS")
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

            # Train on all datasets
            all_results = {}  # {dataset: {model_name: accuracy}}
            for ds_name in dataset_storage.keys():
                print(f"\n{'='*60}")
                print(f"Training on {ds_name} dataset")
                print(f"{'='*60}")
                all_results[ds_name] = {}

                for arch_name, hidden_layers in architectures_to_train:
                    print(f"\n  Training {hidden_layers}...")
                    model = MLP(hidden_layers)
                    trained_model, train_losses, val_losses, test_losses = train_model(
                        model, train_loaders[ds_name], val_loaders[ds_name], test_loaders[ds_name],
                        epochs=epochs, lr=lr, verbose=False
                    )

                    trained_models[ds_name][arch_name] = trained_model
                    train_histories[ds_name][arch_name] = {
                        "train_losses": train_losses,
                        "val_losses": val_losses,
                        "test_losses": test_losses
                    }

                    final_acc = evaluate_model(trained_model, test_loaders[ds_name])
                    all_results[ds_name][arch_name] = final_acc
                    print(f"    Test Accuracy: {final_acc*100:.2f}%")

            # Display summary comparison
            print("\n" + "="*60)
            print("TRAINING SUMMARY - ALL DATASETS")
            print("="*60)
            print(f"{'Architecture':24s} | ", end="")
            for ds_name in dataset_storage.keys():
                print(f"{ds_name:^10s} | ", end="")
            print()
            print("-"*70)

            for arch_name, hidden_layers in architectures_to_train:
                print(f"{str(hidden_layers):24s} | ", end="")
                for ds_name in dataset_storage.keys():
                    acc = all_results[ds_name][arch_name]
                    print(f"{acc*100:9.2f}% | ", end="")
                print()

            for ds_name, histories in train_histories.items():
                if histories:
                    plot_learning_curves(histories, title=f"Learning Curves - {ds_name}")

            # Save to file
            with open('test_results.txt', 'a') as f:
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                f.write(f"\n{'='*60}\n")
                f.write(f"Multi-Dataset Training Results - {timestamp}\n")
                f.write(f"{'='*60}\n")
                f.write(f"{'Architecture':24s} | ")
                for ds_name in dataset_storage.keys():
                    f.write(f"{ds_name:^10s} | ")
                f.write("\n" + "-"*70 + "\n")
                for arch_name, hidden_layers in architectures_to_train:
                    f.write(f"{str(hidden_layers):24s} | ")
                    for ds_name in dataset_storage.keys():
                        acc = all_results[ds_name][arch_name]
                        f.write(f"{acc*100:9.2f}% | ")
                    f.write("\n")

            print("\nResults saved to test_results.txt")

        elif choice == "5" and len(dataset_storage) > 1:
            if not any(any(models.values()) for models in trained_models.values()):
                print("\nNo trained models yet. Please train models first (option 4).")
                continue

            print("\n" + "="*60)
            print("CROSS-DATASET COMPARISON")
            print("="*60)
            print("This tests models trained on one dataset against all datasets")
            print()

            # Find which datasets have trained models
            available_sources = [ds for ds in dataset_storage.keys() if trained_models[ds]]
            if not available_sources:
                print("No trained models available.")
                continue

            print("Select source dataset (models trained on):")
            for i, ds_name in enumerate(available_sources, 1):
                num_models = len(trained_models[ds_name])
                print(f"  {i}. {ds_name} ({num_models} models)")

            source_choice = input(f"Choice (1-{len(available_sources)}): ").strip()
            if not source_choice.isdigit() or not (1 <= int(source_choice) <= len(available_sources)):
                print("Invalid choice.")
                continue

            source_ds = available_sources[int(source_choice) - 1]
            available_models = list(trained_models[source_ds].keys())

            if not available_models:
                print(f"No models trained on {source_ds} dataset.")
                continue

            print(f"\nTesting models trained on {source_ds} across all datasets...")
            print("\n" + "="*60)
            print("CROSS-DATASET PERFORMANCE")
            print("="*60)
            print(f"{'Architecture':24s} | ", end="")
            for test_ds in dataset_storage.keys():
                print(f"{test_ds:^10s} | ", end="")
            print()
            print("-"*70)

            cross_results = {}  # {model_name: {test_dataset: accuracy}}
            for model_name in available_models:
                # Get the hidden layers for this model from MLP_ARCHITECTURES
                hidden_layers = MLP_ARCHITECTURES.get(model_name, "N/A")
                cross_results[model_name] = {}
                print(f"{str(hidden_layers):24s} | ", end="")
                model = trained_models[source_ds][model_name]

                for test_ds in dataset_storage.keys():
                    acc = evaluate_model(model, test_loaders[test_ds])
                    cross_results[model_name][test_ds] = acc
                    print(f"{acc*100:9.2f}% | ", end="")
                print()

            # Save to file
            with open('test_results.txt', 'a') as f:
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                f.write(f"\n{'='*60}\n")
                f.write(f"Cross-Dataset Comparison - {timestamp}\n")
                f.write(f"Models trained on: {source_ds}\n")
                f.write(f"{'='*60}\n")
                f.write(f"{'Architecture':24s} | ")
                for test_ds in dataset_storage.keys():
                    f.write(f"{test_ds:^10s} | ")
                f.write("\n" + "-"*70 + "\n")
                for model_name in available_models:
                    # Get the hidden layers for this model
                    hidden_layers = MLP_ARCHITECTURES.get(model_name, "N/A")
                    f.write(f"{str(hidden_layers):24s} | ")
                    for test_ds in dataset_storage.keys():
                        acc = cross_results[model_name][test_ds]
                        f.write(f"{acc*100:9.2f}% | ")
                    f.write("\n")

            print("\nResults saved to test_results.txt")

        elif choice == "0":
            print("\nExiting. Goodbye!")
            break

        elif choice == "6":
            print("\n" + "="*60)
            print("CLEANUP")
            print("="*60)

            clear_test = input("Clear test data? (y/n): ").strip().lower() == "y"
            clear_train = input("Clear training data? (y/n): ").strip().lower() == "y"

            if clear_test:
                test_loaders = {ds: None for ds in test_loaders.keys()}
                print("Test data cleared.")
            if clear_train:
                trained_models = {ds: {} for ds in trained_models.keys()}
                train_histories = {ds: {} for ds in train_histories.keys()}
                print("Training data cleared.")

            results_path = Path("test_results.txt")
            if results_path.exists():
                keep_str = input("Leave how many last lines? (default 1, max 100): ").strip()
                keep_lines = 1
                if keep_str:
                    try:
                        keep_lines = int(keep_str)
                    except ValueError:
                        keep_lines = 1
                keep_lines = max(1, min(100, keep_lines))

                lines = results_path.read_text(encoding="utf-8").splitlines()
                remaining = lines[-keep_lines:] if lines else []
                results_path.write_text("\n".join(remaining) + ("\n" if remaining else ""), encoding="utf-8")
                print(f"test_results.txt trimmed to last {keep_lines} lines.")
            else:
                print("test_results.txt not found; nothing to trim.")

        else:
            max_valid = 6 if len(dataset_storage) > 1 else 4
            print(f"Invalid option. Please select 0-{max_valid}.")


if __name__ == "__main__":
    main()
