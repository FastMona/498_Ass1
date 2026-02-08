"""Compatibility wrapper for the refactored modules."""

from dash import main
from data import DATA_PARAMS, generate_intertwined_spirals
from MLPx6 import (
    MLP,
    MLP_ARCHITECTURES,
    prepare_data,
    train_model,
    evaluate_model,
    predict,
    plot_learning_curves,
)

__all__ = [
    "DATA_PARAMS",
    "generate_intertwined_spirals",
    "MLP_ARCHITECTURES",
    "MLP",
    "prepare_data",
    "train_model",
    "evaluate_model",
    "predict",
    "plot_learning_curves",
    "main",
]


if __name__ == "__main__":
    main()
