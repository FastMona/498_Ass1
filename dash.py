import matplotlib.pyplot as plt
import torch
from pathlib import Path
from datetime import datetime

from data import (
    DATA_PARAMS,
    generate_intertwined_spirals,
    normalize_spirals,
    plot_three_datasets,
    plot_three_datasets_with_fp_fn,
)
from MLPx6 import (
    MLP,
    MLP_ARCHITECTURES,
    prepare_data,
    train_model,
    evaluate_model,
    predict,
    confusion_counts,
)


def _generate_datasets(data_params, active_dataset):
    sampling_method = "ALL"
    data_params["sampling_method"] = sampling_method
    dataset_storage = {}
    norm_stats_by_dataset = {}

    if sampling_method == "ALL":
        result = generate_intertwined_spirals(
            n=data_params["n"],
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

    if data_params.get("normalize", False):
        for ds_name, ds_data in list(dataset_storage.items()):
            normalized_data, stats = normalize_spirals(ds_data)
            dataset_storage[ds_name] = normalized_data
            norm_stats_by_dataset[ds_name] = stats
        spirals = dataset_storage[active_dataset]

    return dataset_storage, active_dataset, spirals, sampling_method, norm_stats_by_dataset


def _init_dataloaders(dataset_storage, batch_size=32, test_split=0.2, val_split=0.2):
    train_loaders = {}
    val_loaders = {}
    test_loaders = {}
    trained_models = {}
    train_histories = {}

    for ds_name in dataset_storage.keys():
        train_loaders[ds_name], val_loaders[ds_name], test_loaders[ds_name] = prepare_data(
            dataset_storage[ds_name], test_split=test_split, val_split=val_split, batch_size=batch_size
        )
        trained_models[ds_name] = {}
        train_histories[ds_name] = {}

    return train_loaders, val_loaders, test_loaders, trained_models, train_histories


def _collect_fp_fn_points(model, test_loader):
    model.eval()
    fp_points = []
    fn_points = []
    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            outputs = model(x_batch)
            predicted = (outputs > 0.5).float()
            fp_mask = (predicted == 1) & (y_batch == 0)
            fn_mask = (predicted == 0) & (y_batch == 1)

            if fp_mask.any():
                fp_xy = x_batch[fp_mask.squeeze()]
                fp_points.extend([(float(x), float(y)) for x, y in fp_xy.tolist()])
            if fn_mask.any():
                fn_xy = x_batch[fn_mask.squeeze()]
                fn_points.extend([(float(x), float(y)) for x, y in fn_xy.tolist()])
    return fp_points, fn_points


def _print_point_list(label, points, limit):
    print(f"{label} (showing {min(limit, len(points))} of {len(points)}):")
    if not points:
        print("  (none)")
        return
    for x, y in points[:limit]:
        print(f"  ({x:.4f}, {y:.4f})")


def _read_int(prompt, default, min_value=1):
    value_str = input(prompt).strip()
    if not value_str:
        return default
    try:
        value = int(value_str)
        if value < min_value:
            raise ValueError
        return value
    except ValueError:
        print(f"Invalid value. Using default {default}.")
        return default


def _read_float(prompt, default, min_value=1e-12):
    value_str = input(prompt).strip()
    if not value_str:
        return default
    try:
        value = float(value_str)
        if value < min_value:
            raise ValueError
        return value
    except ValueError:
        print(f"Invalid value. Using default {default}.")
        return default


def _select_architectures():
    print("Available architectures:")
    for i, (name, arch) in enumerate(MLP_ARCHITECTURES.items(), 1):
        print(f"  {i}. {name}: {arch}")

    train_choice = input("\nSelect architectures (default 'all', or '1', '1,2,3'): ").strip().lower()
    if train_choice == "all" or train_choice == "":
        return list(MLP_ARCHITECTURES.items())

    try:
        indices = [int(x.strip()) - 1 for x in train_choice.split(",")]
        return [
            (name, arch) for i, (name, arch) in enumerate(MLP_ARCHITECTURES.items())
            if i in indices
        ]
    except (ValueError, IndexError):
        print("Invalid input. Using all architectures.")
        return list(MLP_ARCHITECTURES.items())


def _print_training_time_summary(train_histories, dataset_storage, points_per_cat):
    ordered_datasets = [name for name in ("RND", "CTR", "EDGE") if name in dataset_storage]
    if not ordered_datasets or not any(train_histories.get(ds) for ds in ordered_datasets):
        print("\nNo training time data available. Train models first (option 3).")
        return

    print("\n" + "="*60)
    print(f"TRAINING TIME SUMMARY - {points_per_cat} POINTS PER CAT")
    print("="*60)

    model_names = [
        name for name in MLP_ARCHITECTURES.keys()
        if any(name in train_histories.get(ds, {}) for ds in ordered_datasets)
    ]
    if not model_names:
        print("No recorded training times found.")
        return

    model_col_width = 34
    time_col_width = 10
    header_parts = [f"{'Model (arch)':{model_col_width}s}"]
    for ds_name in ordered_datasets:
        header_parts.append(f"{ds_name} s".center(time_col_width))
    header_parts.append("avg s".center(time_col_width))
    header_line = " | ".join(header_parts)
    print(header_line)
    print("-" * len(header_line))

    for model_name in model_names:
        arch = MLP_ARCHITECTURES.get(model_name, [])
        label = f"{model_name} {arch}"
        if len(label) > model_col_width:
            label = label[:model_col_width - 3] + "..."
        row_parts = [f"{label:{model_col_width}s}"]
        times = []
        for ds_name in ordered_datasets:
            time_s = train_histories.get(ds_name, {}).get(model_name, {}).get("train_time_s")
            if time_s is None:
                row_parts.append(f"{'--':>9s}")
            else:
                row_parts.append(f"{time_s:9.2f}")
                times.append(time_s)
        avg_time = sum(times) / len(times) if times else None
        row_parts.append(f"{avg_time:9.2f}" if avg_time is not None else f"{'--':>9s}")
        print(" | ".join(row_parts))

    with open("test_results.txt", "a") as f:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        f.write(f"\n{'='*60}\n")
        f.write(f"Training Time Summary - {timestamp}\n")
        f.write(f"POINTS PER CAT: {points_per_cat}\n")
        f.write(f"{'='*60}\n")
        f.write(header_line + "\n")
        f.write("-" * len(header_line) + "\n")
        for model_name in model_names:
            arch = MLP_ARCHITECTURES.get(model_name, [])
            label = f"{model_name} {arch}"
            if len(label) > model_col_width:
                label = label[:model_col_width - 3] + "..."
            row_parts = [f"{label:{model_col_width}s}"]
            times = []
            for ds_name in ordered_datasets:
                time_s = train_histories.get(ds_name, {}).get(model_name, {}).get("train_time_s")
                if time_s is None:
                    row_parts.append(f"{'--':>9s}")
                else:
                    row_parts.append(f"{time_s:9.2f}")
                    times.append(time_s)
            avg_time = sum(times) / len(times) if times else None
            row_parts.append(f"{avg_time:9.2f}" if avg_time is not None else f"{'--':>9s}")
            f.write(" | ".join(row_parts) + "\n")

    print("\nTraining time summary saved to test_results.txt")


def main():
    print("="*60)
    print("INTERTWINED SPIRALS CLASSIFICATION WITH MLPs")
    print("="*60)

    # Start with empty datasets; user generates via option 1.
    active_dataset = "RND"
    dataset_storage = {}
    spirals = []
    sampling_method = DATA_PARAMS.get("sampling_method", "ALL")
    norm_stats_by_dataset = {}

    # Data split parameters - defaults: 64% train / 16% val / 20% test
    split_params = {
        "test_split": 0.2,  # 20%
        "val_split": 0.2    # 20% of remaining (16% overall)
    }

    active_params = {k: DATA_PARAMS[k] for k in ("n", "noise_std", "seed", "normalize")}
    print("\nNo datasets loaded. Use option 1 to generate datasets.")
    print(f"Data parameters (active): {active_params}")
    print(f"Sampling method: {sampling_method}")
    
    # Calculate and display split percentages
    test_pct = split_params["test_split"] * 100
    val_pct = split_params["val_split"] * (1 - split_params["test_split"]) * 100
    train_pct = 100 - test_pct - val_pct
    print(f"Data split: {train_pct:.0f}% train / {val_pct:.0f}% validation / {test_pct:.0f}% test")

    # Create models dictionary - organized by dataset
    train_loaders, val_loaders, test_loaders, trained_models, train_histories = _init_dataloaders(
        dataset_storage, batch_size=32, test_split=split_params["test_split"], val_split=split_params["val_split"]
    )

    plt.ion()

    while True:
        print("\n" + "="*60)
        print("MAIN MENU")
        print("="*60)
        print("1. Create Datasets")
        print("2. Display Current Dataset")
        if len(dataset_storage) > 1:
            print("3. Train on ALL Datasets")
        print("4. Single Point Test (classify one point)")
        if len(dataset_storage) > 1:
            print("5. Cross-Dataset Comparison")
        print("6. List/Display FP/FN Points (best models)")
        print("7. Training Summary")
        print("9. Cleanup")
        print("0. Exit")
        print("="*60)

        max_option = "9"
        choice = input(f"Select option (1-{max_option}, 0 to exit): ").strip()

        if choice == "1":
            print("\n" + "="*60)
            print("CREATE DATA SETS")
            print("="*60)
            
            # Configure data split percentages
            test_pct = split_params["test_split"] * 100
            val_pct = split_params["val_split"] * (1 - split_params["test_split"]) * 100
            train_pct = 100 - test_pct - val_pct
            print(f"\nCurrent split: {train_pct:.0f}% train / {val_pct:.0f}% validation / {test_pct:.0f}% test")
            
            configure_split = input("Configure data split? (y/n, default n): ").strip().lower()
            if configure_split == 'y':
                test_input = input(f"Test percentage (default {test_pct:.0f}): ").strip()
                if test_input:
                    try:
                        test_value = float(test_input)
                        if 0 < test_value < 100:
                            split_params["test_split"] = test_value / 100
                        else:
                            print("Invalid value. Must be between 0 and 100. Keeping existing value.")
                    except ValueError:
                        print("Invalid value. Keeping existing test percentage.")
                
                # Recalculate for validation input
                remaining = 100 - (split_params["test_split"] * 100)
                current_val_pct = split_params["val_split"] * (1 - split_params["test_split"]) * 100
                
                val_input = input(f"Validation percentage (default {current_val_pct:.0f}, max {remaining:.0f}): ").strip()
                if val_input:
                    try:
                        val_value = float(val_input)
                        if 0 < val_value < remaining:
                            # val_split is fraction of remaining data after test split
                            split_params["val_split"] = val_value / (100 * (1 - split_params["test_split"]))
                        else:
                            print(f"Invalid value. Must be between 0 and {remaining:.0f}. Keeping existing value.")
                    except ValueError:
                        print("Invalid value. Keeping existing validation percentage.")
                
                # Display final split
                test_pct = split_params["test_split"] * 100
                val_pct = split_params["val_split"] * (1 - split_params["test_split"]) * 100
                train_pct = 100 - test_pct - val_pct
                print(f"Final split: {train_pct:.0f}% train / {val_pct:.0f}% validation / {test_pct:.0f}% test")
            
            n_str = input(f"\nNumber of points per region (default {DATA_PARAMS['n']}): ").strip()
            if n_str:
                try:
                    n_value = int(n_str)
                    if n_value > 0:
                        DATA_PARAMS["n"] = n_value
                    else:
                        print("Invalid value. Keeping existing number of points.")
                except ValueError:
                    print("Invalid value. Keeping existing number of points.")

            normalize_str = input("Normalize to zero mean/unit variance? (y/N): ").strip().lower()
            DATA_PARAMS["normalize"] = normalize_str in {"y", "yes"}

            dataset_storage, active_dataset, spirals, sampling_method, norm_stats_by_dataset = _generate_datasets(
                DATA_PARAMS, active_dataset
            )
            if sampling_method == "ALL":
                # Confirmation of user input choices
                print("\n--- DATASET GENERATION CONFIRMATION ---")
                print(f"Split: {100 - split_params['test_split']*100 - split_params['val_split']*(1-split_params['test_split'])*100:.0f}% train / {split_params['val_split']*(1-split_params['test_split'])*100:.0f}% validation / {split_params['test_split']*100:.0f}% test")
                print(f"Points per region: {DATA_PARAMS['n']}")
                print(f"Normalization: {'enabled' if DATA_PARAMS.get('normalize', False) else 'disabled'}")
                print("Generated RND, CTR, and EDGE datasets")
            if DATA_PARAMS.get("normalize", False):
                print("Normalization: enabled (zero mean/unit variance)")

            total_points = len(spirals)
            points_per_spiral = DATA_PARAMS["n"]
            print(f"Total points: {total_points} ({points_per_spiral} per spiral)")

            train_loaders, val_loaders, test_loaders, trained_models, train_histories = _init_dataloaders(
                dataset_storage, batch_size=32, test_split=split_params["test_split"], val_split=split_params["val_split"]
            )

        elif choice == "2":
            print("\n" + "="*60)
            print("DISPLAY CURRENT DATA SETS")
            print("="*60)
            if not dataset_storage:
                print("No datasets; recreate datasets first (option 1).")
                continue
            
            # Display dataset parameters
            print("\nDataset Parameters:")
            print(f"  Points per region: {DATA_PARAMS['n']}")
            print(f"  Noise std: {DATA_PARAMS['noise_std']}")
            print(f"  Seed: {DATA_PARAMS['seed']}")
            print(f"  Sampling method: {DATA_PARAMS.get('sampling_method', 'ALL')}")
            
            # Display split information
            test_pct = split_params["test_split"] * 100
            val_pct = split_params["val_split"] * (1 - split_params["test_split"]) * 100
            train_pct = 100 - test_pct - val_pct
            print(f"  Data split: {train_pct:.0f}% train / {val_pct:.0f}% validation / {test_pct:.0f}% test")
            
            # Display dataset sizes
            print("\nDataset Sizes:")
            for ds_name in dataset_storage.keys():
                total = len(dataset_storage[ds_name])
                n_test = int(total * split_params["test_split"])
                n_trainval = total - n_test
                n_val = int(n_trainval * split_params["val_split"])
                n_train = n_trainval - n_val
                print(f"  {ds_name}: {total} total ({n_train} train, {n_val} val, {n_test} test)")
            
            if not plot_three_datasets(dataset_storage, DATA_PARAMS, norm_stats_by_dataset=norm_stats_by_dataset):
                print("Missing datasets; recreate datasets first (option 1).")
        elif choice == "3" and len(dataset_storage) > 1:
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


            optimizer_type_str = input("Optimizer (1: Adam [default], 2: SGD): ").strip().lower()
            optimizer_type_map = {"1": "adam", "2": "sgd", "adam": "adam", "sgd": "sgd", "": "adam"}
            if optimizer_type_str not in optimizer_type_map:
                print("Invalid optimizer. Using Adam.")
                optimizer_type = "adam"
            else:
                optimizer_type = optimizer_type_map[optimizer_type_str]

            if optimizer_type == "adam":
                lr_default = 0.001
            else:
                lr_default = 0.01

            lr_str = input(f"Learning rate (default {lr_default}): ").strip()
            lr = float(lr_str) if lr_str else lr_default
            print(f"Selected optimizer: {optimizer_type.upper()} | Learning rate: {lr}")

            activation_str = input(
                "Hidden activation (1: relu, 2: leaky_relu, 3: tanh; default 1): "
            ).strip().lower()
            activation_map = {
                "1": "relu",
                "2": "leaky_relu",
                "3": "tanh",
                "relu": "relu",
                "leaky_relu": "leaky_relu",
                "leaky relu": "leaky_relu",
                "tanh": "tanh",
                "": "",
            }
            if activation_str not in activation_map:
                print("Invalid activation. Using relu.")
                activation = "relu"
            else:
                activation = activation_map[activation_str] or "relu"
            print(f"Using activation: {activation}")

            patience_str = input("Early stopping patience (default 3): ").strip()
            patience = int(patience_str) if patience_str else 3

            # Train on all datasets
            all_results = {}  # {dataset: {model_name: accuracy}}
            for ds_name in dataset_storage.keys():
                print(f"\n{'='*60}")
                print(f"Training on {ds_name} dataset")
                print(f"{'='*60}")
                all_results[ds_name] = {}

                for arch_name, hidden_layers in architectures_to_train:
                    print(f"\n  Training {hidden_layers}...")
                    model = MLP(hidden_layers, activation=activation)
                    trained_model, train_losses, val_losses, test_losses, train_time_ms = train_model(
                        model, train_loaders[ds_name], val_loaders[ds_name], test_loaders[ds_name],
                        epochs=epochs, lr=lr, patience=patience, verbose=True, optimizer_type=optimizer_type
                    )
                    train_time_s = train_time_ms / 1000.0

                    trained_models[ds_name][arch_name] = trained_model
                    train_histories[ds_name][arch_name] = {
                        "train_losses": train_losses,
                        "val_losses": val_losses,
                        "test_losses": test_losses,
                        "train_time_s": train_time_s,
                        "accuracy": None,  # will be set below
                    }

                    total_params = sum(param.numel() for param in trained_model.parameters())
                    print(f"    Total parameters saved: {total_params}")
                    tn, fp, fn, tp = confusion_counts(trained_model, test_loaders[ds_name])
                    print("    Confusion (test, best model):")
                    print(f"      TN: {tn:5d}  FP: {fp:5d}")
                    print(f"      FN: {fn:5d}  TP: {tp:5d}")
                    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
                    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
                    print(
                        f"      Precision: {precision:.3f}  Recall: {recall:.3f}  "
                        f"Specificity: {specificity:.3f}  F1: {f1:.3f}"
                    )
                    final_acc = evaluate_model(trained_model, test_loaders[ds_name])
                    all_results[ds_name][arch_name] = final_acc
                    train_histories[ds_name][arch_name]["accuracy"] = final_acc
                    print(f"    Test Accuracy: {final_acc*100:.2f}%")
                    print(f"    Training time: {train_time_s:.2f} s")

            # Display summary comparison
            points_per_cat = DATA_PARAMS.get("n", "?")
            total_points = 2 * points_per_cat if isinstance(points_per_cat, int) else "?"
            print("\n" + "="*60)
            print(f"TRAINING SUMMARY - TOTAL POINTS: {total_points}, {activation.upper()} ACTIVATION")
            print("="*60)

            arch_col_width = 24
            acc_col_width = 10

            header_parts = [f"{'Architecture':{arch_col_width}s}"]
            for ds_name in dataset_storage.keys():
                header_parts.append(f"{ds_name} %".center(acc_col_width))
            header_line = " | ".join(header_parts)
            print(header_line)
            print("-" * len(header_line))

            for arch_name, hidden_layers in architectures_to_train:
                row_parts = [f"{str(hidden_layers):{arch_col_width}s}"]
                for ds_name in dataset_storage.keys():
                    acc = all_results[ds_name][arch_name]
                    row_parts.append(f"{acc * 100:9.2f}%")
                print(" | ".join(row_parts))

            ordered_datasets = [name for name in ("RND", "CTR", "EDGE") if name in train_histories]
            if ordered_datasets:
                fig, axes = plt.subplots(1, len(ordered_datasets), figsize=(6 * len(ordered_datasets), 4))
                if len(ordered_datasets) == 1:
                    axes = [axes]

                all_val_losses = []
                for ds_name in ordered_datasets:
                    histories = train_histories.get(ds_name, {})
                    for history in histories.values():
                        val_losses = history.get("val_losses", [])
                        if val_losses:
                            all_val_losses.extend(val_losses)

                y_min = min(all_val_losses) if all_val_losses else None
                y_max = max(all_val_losses) if all_val_losses else None
                if y_min is not None and y_max is not None:
                    if y_min == y_max:
                        y_min -= 0.01
                        y_max += 0.01
                    padding = 0.05 * (y_max - y_min)
                    y_min -= padding
                    y_max += padding

                for ax, ds_name in zip(axes, ordered_datasets):
                    histories = train_histories.get(ds_name, {})
                    if not histories:
                        continue
                    for model_name, history in histories.items():
                        val_losses = history.get("val_losses", [])
                        if not val_losses:
                            continue
                        epochs_range = range(1, len(val_losses) + 1)
                        arch = MLP_ARCHITECTURES.get(model_name, [])
                        ax.plot(epochs_range, val_losses, label=f"{arch}")
                    ax.set_title(f"Validation Loss - {ds_name}")
                    ax.set_xlabel("Epoch")
                    ax.set_ylabel("Loss")
                    ax.grid(True, alpha=0.3)
                    ax.legend(fontsize=8)
                    if y_min is not None and y_max is not None:
                        ax.set_ylim(y_min, y_max)
                fig.tight_layout()
                plt.show(block=False)
                plt.pause(0.001)

            # Save to file
            with open('test_results.txt', 'a') as f:
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                f.write(f"\n{'='*60}\n")
                points_per_cat = DATA_PARAMS.get("n", "?")
                total_points = 2 * points_per_cat if isinstance(points_per_cat, int) else "?"
                f.write(f"TRAINING SUMMARY - TOTAL POINTS: {total_points}, {activation.upper()} ACTIVATION - {timestamp}\n")
                f.write(f"{'='*60}\n")
                arch_col_width = 24
                acc_col_width = 10

                header_parts = [f"{'Architecture':{arch_col_width}s}"]
                for ds_name in dataset_storage.keys():
                    header_parts.append(f"{ds_name} %".center(acc_col_width))
                header_line = " | ".join(header_parts)
                f.write(header_line + "\n")
                f.write("-" * len(header_line) + "\n")
                for arch_name, hidden_layers in architectures_to_train:
                    row_parts = [f"{str(hidden_layers):{arch_col_width}s}"]
                    for ds_name in dataset_storage.keys():
                        acc = all_results[ds_name][arch_name]
                        row_parts.append(f"{acc * 100:9.2f}%")
                    f.write(" | ".join(row_parts) + "\n")
                    # Remove P/R/S/F1 and duplicate train time row from file output

            print("\nResults saved to test_results.txt")

        elif choice == "4":
            if not any(any(models.values()) for models in trained_models.values()):
                print("\nNo trained models yet. Please train models first (option 3).")
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

        elif choice == "0":
            print("\nExiting. Goodbye!")
            break

        elif choice == "5" and len(dataset_storage) > 1:
            if not any(any(models.values()) for models in trained_models.values()):
                print("\nNo trained models yet. Please train models first (option 3).")
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
            print(f"CROSS PERFORMANCE USING {source_ds} DATASET")
            print("="*60)
            print(f"{'Architecture':24s} | ", end="")
            for test_ds in dataset_storage.keys():
                print(f"{test_ds:^10s} | ", end="")
            print()
            print("-"*70)

            cross_results = {}  # {model_name: {test_dataset: accuracy}}
            for model_name in available_models:
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
                    hidden_layers = MLP_ARCHITECTURES.get(model_name, "N/A")
                    f.write(f"{str(hidden_layers):24s} | ")
                    for test_ds in dataset_storage.keys():
                        acc = cross_results[model_name][test_ds]
                        f.write(f"{acc*100:9.2f}% | ")
                    f.write("\n")

            print("\nResults saved to test_results.txt")

        elif choice == "6":
            if not any(any(models.values()) for models in trained_models.values()):
                print("\nNo trained models yet. Please train models first (option 3).")
                continue

            if not dataset_storage:
                print("\nNo datasets; recreate datasets first (option 1).")
                continue

            limit_str = input("Max points to list per category (default 20): ").strip()
            limit = 20
            if limit_str:
                try:
                    limit = max(1, int(limit_str))
                except ValueError:
                    limit = 20

            fp_fn_by_dataset = {}

            print("\n" + "="*60)
            print("FP/FN POINTS FOR BEST MODEL PER DATASET")
            print("="*60)

            for ds_name in dataset_storage.keys():
                ds_models = trained_models.get(ds_name, {})
                if not ds_models:
                    continue

                best_model_name = None
                best_acc = -1.0
                for model_name, model in ds_models.items():
                    acc = evaluate_model(model, test_loaders[ds_name])
                    if acc > best_acc:
                        best_acc = acc
                        best_model_name = model_name

                if best_model_name is None:
                    continue

                model = ds_models[best_model_name]
                fp_points, fn_points = _collect_fp_fn_points(model, test_loaders[ds_name])
                arch = MLP_ARCHITECTURES.get(best_model_name, [])
                fp_fn_by_dataset[ds_name] = {
                    "fp": fp_points,
                    "fn": fn_points,
                    "model_name": best_model_name,
                    "acc": best_acc,
                }

                print(f"\nDataset: {ds_name}")
                print(f"Best model: {best_model_name} {arch} | acc: {best_acc*100:.2f}%")
                _print_point_list("FP (pred C2, true C1)", fp_points, limit)
                _print_point_list("FN (pred C1, true C2)", fn_points, limit)

            if fp_fn_by_dataset:
                plot_three_datasets_with_fp_fn(
                    dataset_storage,
                    fp_fn_by_dataset,
                    norm_stats_by_dataset=norm_stats_by_dataset,
                )

        elif choice == "7":
            points_per_cat = DATA_PARAMS.get("n", "?")
            ordered_datasets = [name for name in ("RND", "CTR", "EDGE") if name in dataset_storage]
            model_names = [
                name for name in MLP_ARCHITECTURES.keys()
                if any(name in train_histories.get(ds, {}) for ds in ordered_datasets)
            ]
            if not model_names:
                print("No recorded training times found.")
                continue

            # 1. Accuracy Table
            print("\n" + "="*60)
            print(f"TRAINING SUMMARY - ACCURACY (%), {points_per_cat} POINTS PER CAT")
            print("="*60)
            arch_col_width = 24
            acc_col_width = 10
            header_parts = [f"{'Architecture':{arch_col_width}s}"]
            for ds_name in ordered_datasets:
                header_parts.append(f"{ds_name} %".center(acc_col_width))
            header_line = " | ".join(header_parts)
            print(header_line)
            print("-" * len(header_line))
            accuracy_table_lines = [header_line, "-" * len(header_line)]
            for model_name in model_names:
                arch = MLP_ARCHITECTURES.get(model_name, [])
                row_parts = [f"{str(arch):{arch_col_width}s}"]
                for ds_name in ordered_datasets:
                    acc = train_histories.get(ds_name, {}).get(model_name, {}).get("accuracy")
                    row_parts.append(f"{acc * 100:9.2f}%" if acc is not None else f"{'--':>9s}")
                line = " | ".join(row_parts)
                print(line)
                accuracy_table_lines.append(line)

            # 2. CPU Time Table
            print("\n" + "="*60)
            print(f"TRAINING TIME SUMMARY - {points_per_cat} POINTS PER CAT")
            print("="*60)
            model_col_width = 34
            time_col_width = 10
            header_parts = [f"{'Model (arch)':{model_col_width}s}"]
            for ds_name in ordered_datasets:
                header_parts.append(f"{ds_name} s".center(time_col_width))
            header_parts.append("avg s".center(time_col_width))
            header_line = " | ".join(header_parts)
            print(header_line)
            print("-" * len(header_line))
            cpu_time_table_lines = [header_line, "-" * len(header_line)]
            for model_name in model_names:
                arch = MLP_ARCHITECTURES.get(model_name, [])
                label = f"{model_name} {arch}"
                if len(label) > model_col_width:
                    label = label[:model_col_width - 3] + "..."
                row_parts = [f"{label:{model_col_width}s}"]
                times = []
                for ds_name in ordered_datasets:
                    time_s = train_histories.get(ds_name, {}).get(model_name, {}).get("train_time_s")
                    if time_s is None:
                        row_parts.append(f"{'--':>9s}")
                    else:
                        row_parts.append(f"{time_s:9.2f}")
                        times.append(time_s)
                avg_time = sum(times) / len(times) if times else None
                row_parts.append(f"{avg_time:9.2f}" if avg_time is not None else f"{'--':>9s}")
                line = " | ".join(row_parts)
                print(line)
                cpu_time_table_lines.append(line)

            # 3. P/R/S/F1 Table
            print("\n" + "="*60)
            print(f"PRECISION / RECALL / SPECIFICITY / F1 SUMMARY - {points_per_cat} POINTS PER CAT")
            print("="*60)
            header_parts = [f"{'Architecture':{arch_col_width}s}"]
            for ds_name in ordered_datasets:
                header_parts.append(f"{ds_name} P/R/S/F1".center(24))
            header_line = " | ".join(header_parts)
            print(header_line)
            print("-" * len(header_line))
            prsf1_table_lines = [header_line, "-" * len(header_line)]
            for model_name in model_names:
                arch = MLP_ARCHITECTURES.get(model_name, [])
                row_parts = [f"{str(arch):{arch_col_width}s}"]
                for ds_name in ordered_datasets:
                    model = trained_models.get(ds_name, {}).get(model_name)
                    if model is None:
                        row_parts.append(f"{'--':>23s}")
                        continue
                    tn, fp, fn, tp = confusion_counts(model, test_loaders[ds_name])
                    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
                    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
                    row_parts.append(f"{precision:.3f}/{recall:.3f}/{specificity:.3f}/{f1:.3f}".center(24))
                line = " | ".join(row_parts)
                print(line)
                prsf1_table_lines.append(line)

            # Write all tables to test_results.txt
            with open('test_results.txt', 'a') as f:
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                f.write(f"\n{'='*60}\n")
                f.write(f"TRAINING SUMMARY - {timestamp}\n")
                f.write(f"{'='*60}\n")
                f.write("\nACCURACY TABLE\n")
                for line in accuracy_table_lines:
                    f.write(line + "\n")
                f.write("\nCPU TIME TABLE\n")
                for line in cpu_time_table_lines:
                    f.write(line + "\n")
                f.write("\nP/R/S/F1 TABLE\n")
                for line in prsf1_table_lines:
                    f.write(line + "\n")

        elif choice == "9":
            print("\n" + "="*60)
            print("CLEANUP")
            print("="*60)

            clear_test = input("Clear test data? (y/n): ").strip().lower() == "y"
            clear_train = input("Clear training data? (y/n): ").strip().lower() == "y"

            if clear_test:
                dataset_storage = {}
                active_dataset = "RND"
                spirals = []
                train_loaders = {}
                val_loaders = {}
                test_loaders = {}
                trained_models = {}
                train_histories = {}
                plt.close("all")
                print("Datasets cleared.")
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
            print("Invalid option. Please select 0-9.")


if __name__ == "__main__":
    main()
