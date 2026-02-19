import matplotlib.pyplot as plt
import torch
import numpy as np
import json
import sys
from pathlib import Path
from datetime import datetime

from data import (
    DATA_PARAMS,
    generate_interlocked_region_data,
    plot_three_datasets,
    plot_three_datasets_with_fp_fn,
)
from MLPx6 import (
    MLP,
    MLP_ARCHITECTURES,
    get_best_device,
    prepare_data,
    train_model as train_mlp_model,
    evaluate_model as evaluate_mlp_model,
    predict as predict_mlp,
    confusion_counts as confusion_counts_mlp,
)
from CPNx6 import (
    CPN,
    CPN_ARCHITECTURES,
    CPN_PARAMS,
    train_model as train_cpn_model,
    evaluate_model as evaluate_cpn_model,
    predict as predict_cpn,
    confusion_counts as confusion_counts_cpn,
)


DATA_CACHE_PATH = Path("saved_datasets.json")
TRAINING_CACHE_PATH = Path("saved_trainingsets.json")

ALL_ARCHITECTURES = {
    **MLP_ARCHITECTURES,
    **CPN_ARCHITECTURES,
}


def _is_cpn_architecture(model_name):
    return model_name in CPN_ARCHITECTURES


def _format_architecture(model_name):
    if _is_cpn_architecture(model_name):
        arch = CPN_ARCHITECTURES.get(model_name, {})
        return f"K={arch.get('n_kohonen', '?')}, mode={arch.get('input_mode', CPN_PARAMS.get('input_mode', 'augmented_unit_sphere'))}"
    return str(MLP_ARCHITECTURES.get(model_name, []))


def _build_model(model_name, arch_config, activation):
    if _is_cpn_architecture(model_name):
        return CPN(
            n_kohonen=int(arch_config.get("n_kohonen", 100)),
            input_size=int(CPN_PARAMS.get("input_size", 2)),
            output_size=int(CPN_PARAMS.get("output_size", 1)),
            input_mode=str(arch_config.get("input_mode", CPN_PARAMS.get("input_mode", "augmented_unit_sphere"))),
        )
    return MLP(arch_config, activation=activation)


def _train_any_model(model_name, model, train_loader, val_loader, test_loader, epochs, lr, patience, verbose, optimizer_type):
    if _is_cpn_architecture(model_name):
        return train_cpn_model(
            model,
            train_loader,
            val_loader,
            test_loader,
            epochs=epochs,
            patience=patience,
            verbose=verbose,
        )
    return train_mlp_model(
        model,
        train_loader,
        val_loader,
        test_loader,
        epochs=epochs,
        lr=lr,
        patience=patience,
        verbose=verbose,
        optimizer_type=optimizer_type,
    )


def _evaluate_any_model(model_name, model, test_loader):
    if _is_cpn_architecture(model_name):
        return evaluate_cpn_model(model, test_loader)
    return evaluate_mlp_model(model, test_loader)


def _predict_any_model(model_name, model, x, y):
    if _is_cpn_architecture(model_name):
        return predict_cpn(model, x, y)
    return predict_mlp(model, x, y)


def _confusion_any_model(model_name, model, test_loader):
    if _is_cpn_architecture(model_name):
        return confusion_counts_cpn(model, test_loader)
    return confusion_counts_mlp(model, test_loader)


def _generate_datasets(data_params, active_dataset):
    sampling_method = data_params.get("sampling_method", "ALL")
    dataset_storage = {}

    if sampling_method == "ALL":
        result = generate_interlocked_region_data(
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
        data_points = dataset_storage[active_dataset]
    else:
        data_points = generate_interlocked_region_data(**data_params)
        dataset_storage[sampling_method] = data_points
        active_dataset = sampling_method

    return dataset_storage, active_dataset, data_points, sampling_method


def _init_dataloaders(dataset_storage, batch_size=32, test_split=0.2, val_split=0.2):
    train_loaders = {}
    val_loaders = {}
    test_loaders = {}
    trained_models = {}
    train_histories = {}

    for ds_name in dataset_storage.keys():
        if not dataset_storage[ds_name]:
            train_loaders[ds_name] = None
            val_loaders[ds_name] = None
            test_loaders[ds_name] = None
        else:
            train_loaders[ds_name], val_loaders[ds_name], test_loaders[ds_name] = prepare_data(
                dataset_storage[ds_name], test_split=test_split, val_split=val_split, batch_size=batch_size
            )
        trained_models[ds_name] = {}
        train_histories[ds_name] = {}

    return train_loaders, val_loaders, test_loaders, trained_models, train_histories


def _collect_fp_fn_points(model_name, model, test_loader):
    model.eval()
    fp_points = []
    fn_points = []
    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            outputs = model(x_batch)
            if _is_cpn_architecture(model_name):
                probs = outputs
            else:
                probs = torch.sigmoid(outputs)
            predicted = (probs > 0.5).float()
            fp_mask = (predicted == 1) & (y_batch == 0)
            fn_mask = (predicted == 0) & (y_batch == 1)

            if fp_mask.any():
                fp_xy = x_batch[fp_mask.view(-1)]
                if fp_xy.ndim == 1:
                    fp_xy = fp_xy.unsqueeze(0)
                fp_points.extend([(float(x), float(y)) for x, y in fp_xy.tolist()])
            if fn_mask.any():
                fn_xy = x_batch[fn_mask.view(-1)]
                if fn_xy.ndim == 1:
                    fn_xy = fn_xy.unsqueeze(0)
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


def _split_percentages(split_params):
    test_pct = split_params["test_split"] * 100
    val_pct = split_params["val_split"] * (1 - split_params["test_split"]) * 100
    train_pct = 100 - test_pct - val_pct
    return train_pct, val_pct, test_pct


def _compute_prsf1(tn, fp, fn, tp):
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return precision, recall, specificity, f1


def _mean_std_ci(values):
    if not values:
        return 0.0, 0.0, 0.0
    arr = np.asarray(values, dtype=np.float64)
    mean = float(np.mean(arr))
    std = float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0
    ci95 = 1.96 * std / np.sqrt(arr.size) if arr.size > 1 else 0.0
    return mean, std, float(ci95)


def _select_architectures():
    print("Available architectures:")
    for i, (name, arch) in enumerate(ALL_ARCHITECTURES.items(), 1):
        if _is_cpn_architecture(name):
            print(f"  {i}. {name}: K={arch.get('n_kohonen')}, mode={arch.get('input_mode', CPN_PARAMS.get('input_mode', 'augmented_unit_sphere'))}")
        else:
            print(f"  {i}. {name}: {arch}")

    train_choice = input("\nSelect architectures (default 'all', or '1', '1,2,3'): ").strip().lower()
    if train_choice == "all" or train_choice == "":
        return list(ALL_ARCHITECTURES.items())

    try:
        indices = [int(x.strip()) - 1 for x in train_choice.split(",")]
        return [
            (name, arch) for i, (name, arch) in enumerate(ALL_ARCHITECTURES.items())
            if i in indices
        ]
    except (ValueError, IndexError):
        print("Invalid input. Using all architectures.")
        return list(ALL_ARCHITECTURES.items())


def _select_datasets(dataset_storage, prompt_text="Select dataset(s) [RND, CTR, EDGE] (default ALL): "):
    available = [name for name in ("RND", "CTR", "EDGE") if name in dataset_storage]
    if not available:
        return []

    selection = input(prompt_text).strip().upper()
    if not selection or selection == "ALL":
        return available

    chosen = [part.strip() for part in selection.split(",") if part.strip()]
    valid = [name for name in chosen if name in available]
    if not valid:
        print("Invalid dataset selection. Using ALL.")
        return available

    deduped = []
    for name in valid:
        if name not in deduped:
            deduped.append(name)
    return deduped


def _print_training_time_summary(train_histories, dataset_storage, points_per_cat):
    ordered_datasets = [name for name in ("RND", "CTR", "EDGE") if name in dataset_storage]
    if not ordered_datasets or not any(train_histories.get(ds) for ds in ordered_datasets):
        print("\nNo training time data available. Train models first (option 3).")
        return

    print("\n" + "="*60)
    print(f"TRAINING TIME SUMMARY - {points_per_cat} POINTS PER CAT")
    print("="*60)

    model_names = [
        name for name in ALL_ARCHITECTURES.keys()
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
        arch = _format_architecture(model_name)
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
            arch = _format_architecture(model_name)
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


def _save_dataset_cache(dataset_storage, active_dataset, sampling_method, data_params):
    payload = {
        "saved_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "dataset_storage": {
            ds_name: [[float(x), float(y), int(label)] for x, y, label in ds_data]
            for ds_name, ds_data in dataset_storage.items()
        },
        "active_dataset": active_dataset,
        "sampling_method": sampling_method,
        "data_params": {
            key: data_params.get(key)
            for key in ("n", "noise_std", "seed", "sampling_method")
        },
    }
    try:
        DATA_CACHE_PATH.write_text(json.dumps(payload), encoding="utf-8")
        return True
    except OSError:
        return False


def _serialize_model_state(model):
    state_dict = model.state_dict()
    serialized = {}
    for key, value in state_dict.items():
        if isinstance(value, torch.Tensor):
            serialized[key] = value.detach().cpu().numpy().tolist()
        else:
            serialized[key] = np.asarray(value).tolist()
    return serialized


def _save_training_cache(trained_models, train_histories):
    payload = {
        "saved_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "trained_models": {},
    }

    for ds_name, models in trained_models.items():
        if not models:
            continue
        payload["trained_models"][ds_name] = {}

        for model_name, model in models.items():
            history = train_histories.get(ds_name, {}).get(model_name, {})
            payload["trained_models"][ds_name][model_name] = {
                "model_type": "CPN" if _is_cpn_architecture(model_name) else "MLP",
                "architecture": _format_architecture(model_name),
                "state_dict": _serialize_model_state(model),
                "metrics": {
                    "accuracy": float(history["accuracy"]) if history.get("accuracy") is not None else None,
                    "train_time_s": float(history["train_time_s"]) if history.get("train_time_s") is not None else None,
                },
            }

    try:
        TRAINING_CACHE_PATH.write_text(json.dumps(payload), encoding="utf-8")
        return True
    except OSError:
        return False


def _load_dataset_cache():
    if not DATA_CACHE_PATH.exists():
        return None
    try:
        payload = json.loads(DATA_CACHE_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None

    raw_storage = payload.get("dataset_storage", {})
    dataset_storage = {}
    for ds_name, ds_data in raw_storage.items():
        dataset_storage[ds_name] = [
            (float(point[0]), float(point[1]), int(point[2]))
            for point in ds_data
            if isinstance(point, (list, tuple)) and len(point) == 3
        ]

    if not dataset_storage:
        return None

    active_dataset = payload.get("active_dataset", "RND")
    if active_dataset not in dataset_storage:
        active_dataset = next(iter(dataset_storage.keys()))

    sampling_method = payload.get("sampling_method", "ALL")

    data_params = payload.get("data_params", {}) if isinstance(payload.get("data_params", {}), dict) else {}
    saved_at = payload.get("saved_at")

    return {
        "dataset_storage": dataset_storage,
        "active_dataset": active_dataset,
        "sampling_method": sampling_method,
        "data_params": data_params,
        "saved_at": saved_at,
    }


def _delete_dataset_cache():
    if not DATA_CACHE_PATH.exists():
        return False
    try:
        DATA_CACHE_PATH.unlink()
        return True
    except OSError:
        return False


def _delete_training_cache():
    if not TRAINING_CACHE_PATH.exists():
        return False
    try:
        TRAINING_CACHE_PATH.unlink()
        return True
    except OSError:
        return False


def main():
    print("="*60)
    print("INTERLOCKED REGION CLASSIFICATION WITH MLPs + CPNs")
    print("="*60)

    # Start with empty datasets; user generates via option 1.
    active_dataset = "RND"
    dataset_storage = {}
    data_points = []
    sampling_method = DATA_PARAMS.get("sampling_method", "ALL")
    last_train_config = None
    last_mc_summary_rows = None
    last_mc_meta = None

    # Data split parameters - defaults: 64% train / 16% val / 20% test
    split_params = {
        "test_split": 0.2,  # 20%
        "val_split": 0.2    # 20% of remaining (16% overall)
    }

    loaded_cache = _load_dataset_cache()
    if loaded_cache is not None:
        dataset_storage = loaded_cache["dataset_storage"]
        active_dataset = loaded_cache["active_dataset"]
        sampling_method = loaded_cache["sampling_method"]
        for key in ("n", "noise_std", "seed", "sampling_method"):
            if key in loaded_cache["data_params"]:
                DATA_PARAMS[key] = loaded_cache["data_params"][key]
        data_points = dataset_storage.get(active_dataset, [])
        saved_at = loaded_cache.get("saved_at")
        if saved_at:
            print(f"\nLoaded persisted datasets from saved_datasets.json (saved at {saved_at}).")
        else:
            print("\nLoaded persisted datasets from saved_datasets.json.")
    else:
        print("\nNo datasets loaded. Use option 1 to generate datasets.")

    active_params = {k: DATA_PARAMS[k] for k in ("n", "noise_std", "seed")}
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
        print("3. Train on ALL Datasets")
        print("4. Single Point Test (classify one point)")
        print("5. Cross-Dataset Comparison")
        print("6. List/Display FP/FN Points (best models)")
        print("7. Training Summary")
        print("8. Monte Carlo Training (repeat option 3 config)")
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
            train_pct, val_pct, test_pct = _split_percentages(split_params)
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
                train_pct, val_pct, test_pct = _split_percentages(split_params)
                print(f"Final split: {train_pct:.0f}% train / {val_pct:.0f}% validation / {test_pct:.0f}% test")
            
            DATA_PARAMS["n"] = _read_int(
                f"\nNumber of points per region (default {DATA_PARAMS['n']}): ",
                default=DATA_PARAMS["n"],
                min_value=0,
            )

            dataset_storage, active_dataset, data_points, sampling_method = _generate_datasets(
                DATA_PARAMS, active_dataset
            )
            if _save_dataset_cache(dataset_storage, active_dataset, sampling_method, DATA_PARAMS):
                print(
                    "Datasets saved to saved_datasets.json "
                    f"at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} "
                    "(overwrites previous saved datasets)."
                )
            else:
                print("Warning: could not save datasets to disk.")
            last_mc_summary_rows = None
            last_mc_meta = None
            if sampling_method == "ALL":
                # Confirmation of user input choices
                print("\n--- DATASET GENERATION CONFIRMATION ---")
                train_pct, val_pct, test_pct = _split_percentages(split_params)
                print(f"Split: {train_pct:.0f}% train / {val_pct:.0f}% validation / {test_pct:.0f}% test")
                print(f"Points per region: {DATA_PARAMS['n']}")
                print("Generated RND, CTR, and EDGE datasets")

            total_points = len(data_points)
            points_per_region = DATA_PARAMS["n"]
            print(f"Total points: {total_points} ({points_per_region} per region)")

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
            train_pct, val_pct, test_pct = _split_percentages(split_params)
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
            
            if not plot_three_datasets(dataset_storage, DATA_PARAMS):
                print("Missing datasets; recreate datasets first (option 1).")
        elif choice == "3":
            if len(dataset_storage) <= 1:
                print("\nCreate datasets first (option 1) before training all datasets.")
                continue
            print("\n" + "="*60)
            print("TRAIN ON ALL DATASETS")
            print("="*60)
            print(f"Python: {sys.executable}")
            print(f"Torch: {torch.__version__}")
            print(f"Device: {get_best_device()}")
            architectures_to_train = _select_architectures()

            epochs = _read_int("Number of epochs (default 100): ", default=100, min_value=1)


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

            lr = _read_float(f"Learning rate (default {lr_default}): ", default=lr_default, min_value=1e-8)
            print(f"Selected optimizer: {optimizer_type.upper()} | Learning rate: {lr}")

            activation_str = input(
                "Hidden activation (1: leaky_relu [default], 2: relu, 3: tanh): "
            ).strip().lower()
            activation_map = {
                "1": "leaky_relu",
                "2": "relu",
                "3": "tanh",
                "relu": "relu",
                "leaky_relu": "leaky_relu",
                "leaky relu": "leaky_relu",
                "tanh": "tanh",
                "": "",
            }
            if activation_str not in activation_map:
                print("Invalid activation. Using leaky_relu.")
                activation = "leaky_relu"
            else:
                activation = activation_map[activation_str] or "leaky_relu"
            print(f"Using activation: {activation}")

            patience = _read_int("Early stopping patience (default 5): ", default=5, min_value=1)
            last_train_config = {
                "architectures_to_train": architectures_to_train,
                "epochs": epochs,
                "optimizer_type": optimizer_type,
                "lr": lr,
                "activation": activation,
                "patience": patience,
            }

            # Train on all datasets
            all_results = {}  # {dataset: {model_name: accuracy}}
            trainable_datasets = []
            for ds_name in dataset_storage.keys():
                if not train_loaders.get(ds_name):
                    print(f"\n{'='*60}")
                    print(f"Skipping {ds_name} dataset (0 samples)")
                    print(f"{'='*60}")
                    continue
                trainable_datasets.append(ds_name)
                print(f"\n{'='*60}")
                print(f"Training on {ds_name} dataset")
                print(f"{'='*60}")
                all_results[ds_name] = {}

                for arch_name, arch_config in architectures_to_train:
                    print(f"\n  Training {arch_name}: {_format_architecture(arch_name)}...")
                    model = _build_model(arch_name, arch_config, activation)
                    trained_model, train_losses, val_losses, test_losses, train_time_ms = _train_any_model(
                        arch_name,
                        model,
                        train_loaders[ds_name],
                        val_loaders[ds_name],
                        test_loaders[ds_name],
                        epochs=epochs,
                        lr=lr,
                        patience=patience,
                        verbose=True,
                        optimizer_type=optimizer_type,
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

                    if _is_cpn_architecture(arch_name):
                        kohonen_weight_count = int(trained_model.kohonen_weights.numel())
                        grossberg_weight_count = int(trained_model.grossberg_weights.numel())
                        print("    Parameters saved:")
                        print(f"      Kohonen weights:  {kohonen_weight_count}")
                        print(f"      Grossberg weights:{grossberg_weight_count}")
                    else:
                        total_params = sum(param.numel() for param in trained_model.parameters())
                        print(f"    Total parameters saved: {total_params}")
                    tn, fp, fn, tp = _confusion_any_model(arch_name, trained_model, test_loaders[ds_name])
                    print("    Confusion (test set):")
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
                    final_acc = _evaluate_any_model(arch_name, trained_model, test_loaders[ds_name])
                    all_results[ds_name][arch_name] = final_acc
                    train_histories[ds_name][arch_name]["accuracy"] = final_acc
                    print(f"    Test Accuracy: {final_acc*100:.2f}%")
                    print(f"    Training time: {train_time_s:.2f} s")

            # Display summary comparison
            points_per_cat = DATA_PARAMS.get("n", "?")
            total_points = 2 * points_per_cat if isinstance(points_per_cat, int) else "?"
            if trainable_datasets:
                print("\n" + "="*60)
                print(f"TRAINING SUMMARY - TOTAL POINTS: {total_points}, {activation.upper()} ACTIVATION")
                print("="*60)

                arch_col_width = 24
                acc_col_width = 10

                header_parts = [f"{'Architecture':{arch_col_width}s}"]
                for ds_name in trainable_datasets:
                    header_parts.append(f"{ds_name} %".center(acc_col_width))
                header_line = " | ".join(header_parts)
                print(header_line)
                print("-" * len(header_line))

                for arch_name, _arch_config in architectures_to_train:
                    row_parts = [f"{_format_architecture(arch_name):{arch_col_width}s}"]
                    for ds_name in trainable_datasets:
                        acc = all_results[ds_name][arch_name]
                        row_parts.append(f"{acc * 100:9.2f}%")
                    print(" | ".join(row_parts))
            else:
                print("\nNo datasets have samples. Training skipped.")

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
                        arch = _format_architecture(model_name)
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
                for arch_name, _arch_config in architectures_to_train:
                    row_parts = [f"{_format_architecture(arch_name):{arch_col_width}s}"]
                    for ds_name in dataset_storage.keys():
                        acc = all_results[ds_name][arch_name]
                        row_parts.append(f"{acc * 100:9.2f}%")
                    f.write(" | ".join(row_parts) + "\n")
                    # Remove P/R/S/F1 and duplicate train time row from file output

            print("\nResults saved to test_results.txt")

            if _save_training_cache(trained_models, train_histories):
                print(
                    "Training weights saved to saved_trainingsets.json "
                    f"at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
                )
            else:
                print("Warning: could not save training weights to saved_trainingsets.json.")

        elif choice == "8":
            if len(dataset_storage) <= 1:
                print("\nCreate datasets first (option 1) before running Monte Carlo training.")
                continue
            if not dataset_storage:
                print("\nNo datasets; recreate datasets first (option 1).")
                continue
            if last_train_config is None:
                print("\nRun option 3 first to set the training configuration.")
                continue

            print("\n" + "="*60)
            print("MONTE CARLO TRAINING (REPEATED RUNS)")
            print("="*60)
            print(f"Python: {sys.executable}")
            print(f"Torch: {torch.__version__}")
            print(f"Device: {get_best_device()}")
            n_runs = _read_int("Number of Monte Carlo runs (default 20): ", default=20, min_value=2)
            base_seed = _read_int("Base random seed (default 42): ", default=42, min_value=0)

            print("\nSelect architectures for Monte Carlo runs:")
            architectures_to_train = _select_architectures()
            if not architectures_to_train:
                print("No architectures selected. Using option-3 architecture selection.")
                architectures_to_train = last_train_config["architectures_to_train"]
            epochs = last_train_config["epochs"]
            optimizer_type = last_train_config["optimizer_type"]
            lr = last_train_config["lr"]
            activation = last_train_config["activation"]
            patience = last_train_config["patience"]

            ordered_datasets = _select_datasets(
                dataset_storage,
                prompt_text="Dataset(s) for Monte Carlo [RND, CTR, EDGE] (default ALL): ",
            )
            ordered_datasets = [name for name in ordered_datasets if dataset_storage.get(name)]
            if not ordered_datasets:
                print("No non-empty datasets found. Generate datasets first (option 1).")
                continue

            print(f"Using option-3 config -> activation: {activation}, optimizer: {optimizer_type}, lr: {lr}, epochs: {epochs}, patience: {patience}")
            print(f"Datasets: {', '.join(ordered_datasets)}")
            print(f"Running {n_runs} Monte Carlo runs starting at seed {base_seed}...")

            mc_results = {
                ds_name: {
                    arch_name: {
                        "accuracy": [],
                        "precision": [],
                        "recall": [],
                        "specificity": [],
                        "f1": [],
                        "train_time_s": [],
                    }
                    for arch_name, _ in architectures_to_train
                }
                for ds_name in ordered_datasets
            }

            for run_idx in range(n_runs):
                run_seed = base_seed + run_idx
                print(f"\nRun {run_idx + 1}/{n_runs} (seed={run_seed})")

                np.random.seed(run_seed)
                torch.manual_seed(run_seed)

                for ds_name in ordered_datasets:
                    run_train_loader, run_val_loader, run_test_loader = prepare_data(
                        dataset_storage[ds_name],
                        test_split=split_params["test_split"],
                        val_split=split_params["val_split"],
                        batch_size=32,
                    )

                    for arch_name, arch_config in architectures_to_train:
                        model = _build_model(arch_name, arch_config, activation)
                        trained_model, _, _, _, train_time_ms = _train_any_model(
                            arch_name,
                            model,
                            run_train_loader,
                            run_val_loader,
                            run_test_loader,
                            epochs=epochs,
                            lr=lr,
                            patience=patience,
                            verbose=False,
                            optimizer_type=optimizer_type,
                        )

                        tn, fp, fn, tp = _confusion_any_model(arch_name, trained_model, run_test_loader)
                        precision, recall, specificity, f1 = _compute_prsf1(tn, fp, fn, tp)
                        acc = _evaluate_any_model(arch_name, trained_model, run_test_loader)

                        metrics = mc_results[ds_name][arch_name]
                        metrics["accuracy"].append(acc)
                        metrics["precision"].append(precision)
                        metrics["recall"].append(recall)
                        metrics["specificity"].append(specificity)
                        metrics["f1"].append(f1)
                        metrics["train_time_s"].append(train_time_ms / 1000.0)

            print("\n" + "="*60)
            print("MONTE CARLO SUMMARY (MEAN ± STD, 95% CI OF MEAN)")
            print("="*60)
            print(f"{'Dataset':7s} | {'Architecture':20s} | {'Acc%':>16s} | {'P':>16s} | {'R':>16s} | {'S':>16s} | {'F1':>16s} | {'Time(s)':>16s}")
            print("-" * 154)

            summary_rows = []
            for ds_name in ordered_datasets:
                for arch_name, _arch_config in architectures_to_train:
                    m = mc_results[ds_name][arch_name]
                    acc_mean, acc_std, acc_ci = _mean_std_ci(m["accuracy"])
                    p_mean, p_std, p_ci = _mean_std_ci(m["precision"])
                    r_mean, r_std, r_ci = _mean_std_ci(m["recall"])
                    s_mean, s_std, s_ci = _mean_std_ci(m["specificity"])
                    f1_mean, f1_std, f1_ci = _mean_std_ci(m["f1"])
                    t_mean, t_std, t_ci = _mean_std_ci(m["train_time_s"])

                    arch_label = _format_architecture(arch_name)
                    if len(arch_label) > 20:
                        arch_label = arch_label[:17] + "..."

                    print(
                        f"{ds_name:7s} | {arch_label:20s} | "
                        f"{acc_mean*100:6.2f}±{acc_std*100:5.2f} | "
                        f"{p_mean:6.3f}±{p_std:5.3f} | "
                        f"{r_mean:6.3f}±{r_std:5.3f} | "
                        f"{s_mean:6.3f}±{s_std:5.3f} | "
                        f"{f1_mean:6.3f}±{f1_std:5.3f} | "
                        f"{t_mean:6.2f}±{t_std:5.2f}"
                    )

                    summary_rows.append({
                        "dataset": ds_name,
                        "architecture": _format_architecture(arch_name),
                        "acc": (acc_mean, acc_std, acc_ci),
                        "p": (p_mean, p_std, p_ci),
                        "r": (r_mean, r_std, r_ci),
                        "s": (s_mean, s_std, s_ci),
                        "f1": (f1_mean, f1_std, f1_ci),
                        "time": (t_mean, t_std, t_ci),
                    })

            with open('test_results.txt', 'a') as f:
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                f.write(f"\n{'='*80}\n")
                f.write(f"MONTE CARLO SUMMARY - {timestamp}\n")
                f.write(f"Runs: {n_runs}, Base seed: {base_seed}\n")
                f.write(
                    f"Config: activation={activation}, optimizer={optimizer_type}, lr={lr}, "
                    f"epochs={epochs}, patience={patience}\n"
                )
                f.write(f"Split: test={split_params['test_split']:.2f}, val={split_params['val_split']:.2f} of trainval\n")
                f.write(f"{'='*80}\n")
                for row in summary_rows:
                    f.write(
                        f"{row['dataset']} {row['architecture']} | "
                        f"Acc={row['acc'][0]*100:.2f}±{row['acc'][1]*100:.2f} (95%CI ±{row['acc'][2]*100:.2f}) | "
                        f"P={row['p'][0]:.3f}±{row['p'][1]:.3f} (95%CI ±{row['p'][2]:.3f}) | "
                        f"R={row['r'][0]:.3f}±{row['r'][1]:.3f} (95%CI ±{row['r'][2]:.3f}) | "
                        f"S={row['s'][0]:.3f}±{row['s'][1]:.3f} (95%CI ±{row['s'][2]:.3f}) | "
                        f"F1={row['f1'][0]:.3f}±{row['f1'][1]:.3f} (95%CI ±{row['f1'][2]:.3f}) | "
                        f"Time={row['time'][0]:.2f}±{row['time'][1]:.2f}s (95%CI ±{row['time'][2]:.2f}s)\n"
                    )

            last_mc_summary_rows = summary_rows
            last_mc_meta = {
                "n_runs": n_runs,
                "base_seed": base_seed,
                "activation": activation,
                "optimizer_type": optimizer_type,
                "lr": lr,
                "epochs": epochs,
                "patience": patience,
            }

            print("\nMonte Carlo summary saved to test_results.txt")

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
                        pred, conf = _predict_any_model(model_name, model, x, y)
                        all_conf_values.append(conf)
                        class_name = "C1" if pred == 0 else "C2"
                        arch = _format_architecture(model_name)
                        label = f"{model_name} {arch}"
                        print(f"{label:30s} -> {class_name} (confidence: {conf:.4f})")
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

        elif choice == "5":
            if len(dataset_storage) <= 1:
                print("\nCreate datasets first (option 1) before cross-dataset comparison.")
                continue
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
                hidden_layers = _format_architecture(model_name)
                cross_results[model_name] = {}
                print(f"{str(hidden_layers):24s} | ", end="")
                model = trained_models[source_ds][model_name]

                for test_ds in dataset_storage.keys():
                    acc = _evaluate_any_model(model_name, model, test_loaders[test_ds])
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
                    hidden_layers = _format_architecture(model_name)
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

            limit = _read_int("Max points to list per category (default 20): ", default=20, min_value=1)

            available_model_names = [
                name for name in ALL_ARCHITECTURES.keys()
                if any(name in trained_models.get(ds_name, {}) for ds_name in dataset_storage.keys())
            ]
            if not available_model_names:
                print("No trained architectures available.")
                continue

            print("\nAvailable architectures for FP/FN:")
            for i, model_name in enumerate(available_model_names, 1):
                print(f"  {i}. {model_name}: {_format_architecture(model_name)}")

            model_choice = input(
                "Select architecture index/name (default: best per dataset): "
            ).strip()

            selected_model_name = None
            if model_choice:
                if model_choice.isdigit():
                    model_index = int(model_choice) - 1
                    if 0 <= model_index < len(available_model_names):
                        selected_model_name = available_model_names[model_index]
                    else:
                        print("Invalid index. Falling back to best per dataset.")
                elif model_choice in available_model_names:
                    selected_model_name = model_choice
                else:
                    print("Unknown architecture. Falling back to best per dataset.")

            fp_fn_by_dataset = {}

            print("\n" + "="*60)
            if selected_model_name is None:
                print("FP/FN POINTS FOR BEST MODEL PER DATASET")
            else:
                print(f"FP/FN POINTS FOR {selected_model_name} PER DATASET")
            print("="*60)

            for ds_name in dataset_storage.keys():
                ds_models = trained_models.get(ds_name, {})
                if not ds_models:
                    continue

                if selected_model_name is None:
                    best_model_name = None
                    best_acc = -1.0
                    for model_name, model in ds_models.items():
                        acc = _evaluate_any_model(model_name, model, test_loaders[ds_name])
                        if acc > best_acc:
                            best_acc = acc
                            best_model_name = model_name

                    if best_model_name is None:
                        continue
                    model_name_for_ds = best_model_name
                    model = ds_models[model_name_for_ds]
                    model_acc = best_acc
                else:
                    if selected_model_name not in ds_models:
                        print(f"\nDataset: {ds_name}")
                        print(f"Model {selected_model_name} not trained on this dataset.")
                        continue
                    model_name_for_ds = selected_model_name
                    model = ds_models[model_name_for_ds]
                    model_acc = _evaluate_any_model(model_name_for_ds, model, test_loaders[ds_name])

                fp_points, fn_points = _collect_fp_fn_points(model_name_for_ds, model, test_loaders[ds_name])
                arch = _format_architecture(model_name_for_ds)
                fp_fn_by_dataset[ds_name] = {
                    "fp": fp_points,
                    "fn": fn_points,
                    "model_name": model_name_for_ds,
                    "acc": model_acc,
                }

                print(f"\nDataset: {ds_name}")
                if selected_model_name is None:
                    print(f"Best model: {model_name_for_ds} {arch} | acc: {model_acc*100:.2f}%")
                else:
                    print(f"Model: {model_name_for_ds} {arch} | acc: {model_acc*100:.2f}%")
                _print_point_list("FP (pred C2, true C1)", fp_points, limit)
                _print_point_list("FN (pred C1, true C2)", fn_points, limit)

            if fp_fn_by_dataset:
                plot_three_datasets_with_fp_fn(
                    dataset_storage,
                    fp_fn_by_dataset,
                )

        elif choice == "7":
            points_per_cat = DATA_PARAMS.get("n", "?")
            ordered_datasets = _select_datasets(
                dataset_storage,
                prompt_text="Dataset(s) for summary [RND, CTR, EDGE] (default ALL): ",
            )
            if not ordered_datasets:
                print("No datasets available. Create datasets first (option 1).")
                continue
            model_names = [
                name for name in ALL_ARCHITECTURES.keys()
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
                arch = _format_architecture(model_name)
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
                arch = _format_architecture(model_name)
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

            # 2b. Parameter Count Table
            print("\n" + "="*60)
            print("PARAMETER COUNT SUMMARY - CURRENT ARCHITECTURES")
            print("="*60)
            param_col_width = 24
            param_header = f"{'Architecture':{arch_col_width}s} | {'Params':>{param_col_width}s}"
            print(param_header)
            print("-" * len(param_header))
            param_table_lines = [param_header, "-" * len(param_header)]
            for model_name in model_names:
                arch = _format_architecture(model_name)
                model_for_params = None
                for ds_name in ordered_datasets:
                    candidate = trained_models.get(ds_name, {}).get(model_name)
                    if candidate is not None:
                        model_for_params = candidate
                        break
                if model_for_params is None:
                    param_text = "N/A"
                else:
                    if _is_cpn_architecture(model_name):
                        kohonen_weight_count = int(model_for_params.kohonen_weights.numel())
                        grossberg_weight_count = int(model_for_params.grossberg_weights.numel())
                        param_text = f"K:{kohonen_weight_count:,} G:{grossberg_weight_count:,}"
                    else:
                        param_text = f"{sum(param.numel() for param in model_for_params.parameters()):,}"
                line = f"{str(arch):{arch_col_width}s} | {param_text:>{param_col_width}s}"
                print(line)
                param_table_lines.append(line)

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
                arch = _format_architecture(model_name)
                row_parts = [f"{str(arch):{arch_col_width}s}"]
                for ds_name in ordered_datasets:
                    model = trained_models.get(ds_name, {}).get(model_name)
                    if model is None:
                        row_parts.append(f"{'--':>23s}")
                        continue
                    tn, fp, fn, tp = _confusion_any_model(model_name, model, test_loaders[ds_name])
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
                f.write("\nPARAMETER COUNT TABLE\n")
                for line in param_table_lines:
                    f.write(line + "\n")
                f.write("\nP/R/S/F1 TABLE\n")
                for line in prsf1_table_lines:
                    f.write(line + "\n")

                if last_mc_summary_rows and last_mc_meta:
                    print("\n" + "="*60)
                    print("MONTE CARLO SNAPSHOT (FROM OPTION 8)")
                    print("="*60)
                    print(
                        f"Runs: {last_mc_meta['n_runs']}, Base seed: {last_mc_meta['base_seed']}, "
                        f"Act: {last_mc_meta['activation']}, Opt: {last_mc_meta['optimizer_type']}, "
                        f"LR: {last_mc_meta['lr']}, Epochs: {last_mc_meta['epochs']}, Patience: {last_mc_meta['patience']}"
                    )
                    print(f"{'Dataset':7s} | {'Architecture':20s} | {'Acc%':>16s} | {'F1':>16s} | {'Time(s)':>16s}")
                    print("-" * 92)
                    for row in last_mc_summary_rows:
                        arch_label = row["architecture"]
                        if len(arch_label) > 20:
                            arch_label = arch_label[:17] + "..."
                        print(
                            f"{row['dataset']:7s} | {arch_label:20s} | "
                            f"{row['acc'][0]*100:6.2f}±{row['acc'][1]*100:5.2f} | "
                            f"{row['f1'][0]:6.3f}±{row['f1'][1]:5.3f} | "
                            f"{row['time'][0]:6.2f}±{row['time'][1]:5.2f}"
                        )

                    f.write("\nMONTE CARLO SNAPSHOT (FROM OPTION 8)\n")
                    f.write(
                        f"Runs: {last_mc_meta['n_runs']}, Base seed: {last_mc_meta['base_seed']}, "
                        f"Act: {last_mc_meta['activation']}, Opt: {last_mc_meta['optimizer_type']}, "
                        f"LR: {last_mc_meta['lr']}, Epochs: {last_mc_meta['epochs']}, Patience: {last_mc_meta['patience']}\n"
                    )
                    f.write(f"{'Dataset':7s} | {'Architecture':20s} | {'Acc%':>16s} | {'F1':>16s} | {'Time(s)':>16s}\n")
                    f.write("-" * 92 + "\n")
                    for row in last_mc_summary_rows:
                        arch_label = row["architecture"]
                        if len(arch_label) > 20:
                            arch_label = arch_label[:17] + "..."
                        f.write(
                            f"{row['dataset']:7s} | {arch_label:20s} | "
                            f"{row['acc'][0]*100:6.2f}±{row['acc'][1]*100:5.2f} | "
                            f"{row['f1'][0]:6.3f}±{row['f1'][1]:5.3f} | "
                            f"{row['time'][0]:6.2f}±{row['time'][1]:5.2f}\n"
                        )

        elif choice == "9":
            print("\n" + "="*60)
            print("CLEANUP")
            print("="*60)

            clear_test = input("Clear test data? (y/n): ").strip().lower() == "y"
            clear_train = input("Clear training data? (y/n): ").strip().lower() == "y"

            if clear_test:
                dataset_storage = {}
                active_dataset = "RND"
                data_points = []
                train_loaders = {}
                val_loaders = {}
                test_loaders = {}
                trained_models = {}
                train_histories = {}
                last_train_config = None
                last_mc_summary_rows = None
                last_mc_meta = None
                plt.close("all")
                if _delete_dataset_cache():
                    print("Persisted datasets removed (saved_datasets.json).")
                if _delete_training_cache():
                    print("Persisted training weights removed (saved_trainingsets.json).")
                print("Datasets cleared.")
                print("Test data cleared.")
            if clear_train:
                trained_models = {ds: {} for ds in trained_models.keys()}
                train_histories = {ds: {} for ds in train_histories.keys()}
                last_train_config = None
                last_mc_summary_rows = None
                last_mc_meta = None
                if _delete_training_cache():
                    print("Persisted training weights removed (saved_trainingsets.json).")
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

                try:
                    lines = results_path.read_text(encoding="utf-8").splitlines()
                except UnicodeDecodeError:
                    try:
                        lines = results_path.read_text(encoding="cp1252").splitlines()
                    except UnicodeDecodeError:
                        lines = results_path.read_text(encoding="utf-8", errors="replace").splitlines()
                remaining = lines[-keep_lines:] if lines else []
                results_path.write_text("\n".join(remaining) + ("\n" if remaining else ""), encoding="utf-8")
                print(f"test_results.txt trimmed to last {keep_lines} lines.")
            else:
                print("test_results.txt not found; nothing to trim.")

        else:
            print("Invalid option. Please select 0-9.")


if __name__ == "__main__":
    main()
