import pickle
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


# ============================
# CPN Architectures (easily modifiable)
# ============================

CPN_ARCHITECTURES = {
	"CPN_1": {"n_kohonen": 100, "input_mode": "l2"},
	"CPN_2": {"n_kohonen": 150, "input_mode": "l2"},
	"CPN_3": {"n_kohonen": 200, "input_mode": "l2"},
	"CPN_4": {"n_kohonen": 100, "input_mode": "l2"},
	"CPN_5": {"n_kohonen": 100, "input_mode": "l2"},
}


# ============================
# CPN Hyperparameters (easily modifiable)
# ============================

CPN_PARAMS = {
	"input_size": 2,
	"output_size": 1,
	"input_mode": "l2",  # "augmented_unit_sphere" or "l2"
	"eta_start": 0.35,
	"eta_end": 0.03,
	"alpha": 0.08,
	"kohonen_ratio": 0.6,
	"patience": 8,
}


def _normalize_rows(array_2d, eps=1e-12):
	norms = np.linalg.norm(array_2d, axis=1, keepdims=True)
	return array_2d / np.clip(norms, eps, None)


def _loader_to_numpy(data_loader):
	x_parts = []
	y_parts = []
	for x_batch, y_batch in data_loader:
		x_parts.append(x_batch.detach().cpu().numpy())
		y_parts.append(y_batch.detach().cpu().numpy())
	x_np = np.concatenate(x_parts, axis=0).astype(np.float32)
	y_np = np.concatenate(y_parts, axis=0).astype(np.float32).reshape(-1)
	return x_np, y_np


class CPN(nn.Module):
	"""
	Counter Propagation Network for binary classification.

	Layer flow:
	  Input -> Kohonen (competitive winner-take-all) -> Grossberg output (binary target mapping)
	"""

	def __init__(self, n_kohonen=100, input_size=2, output_size=1, input_mode="augmented_unit_sphere"):
		super(CPN, self).__init__()

		if n_kohonen <= 0:
			raise ValueError("n_kohonen must be positive")
		if input_size != 2:
			raise ValueError("This CPN implementation expects 2D Cartesian inputs")
		if output_size != 1:
			raise ValueError("This CPN implementation supports binary output only")
		if input_mode not in {"augmented_unit_sphere", "l2"}:
			raise ValueError("input_mode must be 'augmented_unit_sphere' or 'l2'")

		self.n_kohonen = int(n_kohonen)
		self.input_size = int(input_size)
		self.output_size = int(output_size)
		self.input_mode = input_mode

		kohonen_dim = 3 if self.input_mode == "augmented_unit_sphere" else 2
		kohonen_init = np.random.normal(0.0, 1.0, (self.n_kohonen, kohonen_dim)).astype(np.float32)
		kohonen_init = _normalize_rows(kohonen_init)
		grossberg_init = np.random.uniform(0.45, 0.55, (self.n_kohonen, 1)).astype(np.float32)

		self.kohonen_weights = nn.Parameter(torch.from_numpy(kohonen_init), requires_grad=False)
		self.grossberg_weights = nn.Parameter(torch.from_numpy(grossberg_init), requires_grad=False)

	def _preprocess_numpy(self, x_np):
		x = np.asarray(x_np, dtype=np.float32)
		if x.ndim == 1:
			x = x.reshape(1, -1)

		if self.input_mode == "l2":
			x = _normalize_rows(x)
			return x

		radial = np.linalg.norm(x, axis=1, keepdims=True)
		scale = np.maximum(radial, 1.0)
		x_scaled = x / scale
		radius_sq = np.sum(np.square(x_scaled), axis=1, keepdims=True)
		x3 = np.sqrt(np.clip(1.0 - radius_sq, 0.0, 1.0))
		return np.concatenate([x_scaled, x3], axis=1)

	def _preprocess_tensor(self, x_tensor):
		x_np = x_tensor.detach().cpu().numpy()
		x_prep = self._preprocess_numpy(x_np)
		return torch.from_numpy(x_prep.astype(np.float32))

	def winner_indices(self, x_tensor):
		x_prep = self._preprocess_tensor(x_tensor)
		dists = torch.cdist(x_prep, self.kohonen_weights)
		return torch.argmin(dists, dim=1)

	def forward(self, x):
		winners = self.winner_indices(x)
		return self.grossberg_weights[winners]


def train_model(
	model,
	train_loader,
	val_loader,
	test_loader,
	epochs=100,
	lr=None,
	patience=8,
	verbose=True,
	optimizer_type=None,
):
	"""
	Train CPN in two phases:
	  1) Unsupervised Kohonen winner updates
	  2) Supervised Grossberg winner-output updates with validation early stopping
	"""
	del lr, optimizer_type

	x_train, y_train = _loader_to_numpy(train_loader)
	x_val, y_val = _loader_to_numpy(val_loader)
	x_test, y_test = _loader_to_numpy(test_loader)

	x_train_prep = model._preprocess_numpy(x_train)
	x_val_prep = model._preprocess_numpy(x_val)
	x_test_prep = model._preprocess_numpy(x_test)

	kohonen_weights = model.kohonen_weights.detach().cpu().numpy().copy()
	grossberg_weights = model.grossberg_weights.detach().cpu().numpy().copy().reshape(-1)

	eta_start = float(CPN_PARAMS.get("eta_start", 0.35))
	eta_end = float(CPN_PARAMS.get("eta_end", 0.03))
	alpha = float(CPN_PARAMS.get("alpha", 0.08))
	ratio = float(CPN_PARAMS.get("kohonen_ratio", 0.6))

	total_epochs = max(1, int(epochs))
	kohonen_epochs = max(1, min(total_epochs - 1 if total_epochs > 1 else 1, int(round(total_epochs * ratio))))
	grossberg_epochs = max(1, total_epochs - kohonen_epochs)

	train_losses = []
	val_losses = []
	test_losses = []

	best_val_loss = float("inf")
	best_state = None
	best_epoch = 0
	patience_counter = 0
	effective_patience = max(1, int(patience))

	def _predict_with_weights(x_prep, w_k, w_g):
		dists = np.linalg.norm(x_prep[:, None, :] - w_k[None, :, :], axis=2)
		winners = np.argmin(dists, axis=1)
		probs = w_g[winners]
		return probs, winners

	start_time = time.perf_counter()

	if verbose:
		print(f"CPN phase 1 (Kohonen): {kohonen_epochs} epochs")
	for epoch in range(kohonen_epochs):
		if kohonen_epochs == 1:
			eta = eta_end
		else:
			decay = epoch / (kohonen_epochs - 1)
			eta = eta_start * ((eta_end / eta_start) ** decay)

		indices = np.random.permutation(x_train_prep.shape[0])
		for idx in indices:
			x_i = x_train_prep[idx]
			dists = np.linalg.norm(kohonen_weights - x_i, axis=1)
			winner = int(np.argmin(dists))
			kohonen_weights[winner] = kohonen_weights[winner] + eta * (x_i - kohonen_weights[winner])
			kohonen_weights[winner] = _normalize_rows(kohonen_weights[winner:winner + 1])[0]

	if verbose:
		print(f"CPN phase 2 (Grossberg): {grossberg_epochs} epochs")
	for epoch in range(grossberg_epochs):
		indices = np.random.permutation(x_train_prep.shape[0])
		for idx in indices:
			x_i = x_train_prep[idx]
			y_i = y_train[idx]
			dists = np.linalg.norm(kohonen_weights - x_i, axis=1)
			winner = int(np.argmin(dists))
			grossberg_weights[winner] = grossberg_weights[winner] + alpha * (y_i - grossberg_weights[winner])

		train_probs, _ = _predict_with_weights(x_train_prep, kohonen_weights, grossberg_weights)
		val_probs, _ = _predict_with_weights(x_val_prep, kohonen_weights, grossberg_weights)
		test_probs, _ = _predict_with_weights(x_test_prep, kohonen_weights, grossberg_weights)

		train_loss = float(np.mean(np.square(train_probs - y_train)))
		val_loss = float(np.mean(np.square(val_probs - y_val)))
		test_loss = float(np.mean(np.square(test_probs - y_test)))

		train_losses.append(train_loss)
		val_losses.append(val_loss)
		test_losses.append(test_loss)

		if val_loss < best_val_loss:
			best_val_loss = val_loss
			best_state = {
				"kohonen_weights": kohonen_weights.copy(),
				"grossberg_weights": grossberg_weights.copy(),
			}
			best_epoch = epoch
			patience_counter = 0
		else:
			patience_counter += 1

		if verbose and ((epoch + 1) % 10 == 0 or epoch == grossberg_epochs - 1):
			print(
				f"Grossberg Epoch {epoch + 1}/{grossberg_epochs} - "
				f"Train: {train_loss:.4f}, Val: {val_loss:.4f}, Test: {test_loss:.4f}"
			)

		if patience_counter >= effective_patience:
			if verbose:
				print(
					f"CPN early stopping at grossberg epoch {epoch + 1} "
					f"(best: {best_epoch + 1}, val loss: {best_val_loss:.4f})"
				)
			break

	train_time_ms = (time.perf_counter() - start_time) * 1000.0

	if best_state is not None:
		kohonen_weights = best_state["kohonen_weights"]
		grossberg_weights = best_state["grossberg_weights"]
		train_losses = train_losses[:best_epoch + 1]
		val_losses = val_losses[:best_epoch + 1]
		test_losses = test_losses[:best_epoch + 1]

	with torch.no_grad():
		model.kohonen_weights.copy_(torch.from_numpy(kohonen_weights.astype(np.float32)))
		model.grossberg_weights.copy_(torch.from_numpy(grossberg_weights.astype(np.float32).reshape(-1, 1)))

	if verbose:
		print(f"Training time: {train_time_ms / 1000.0:.2f} s")

	return model, train_losses, val_losses, test_losses, train_time_ms


def evaluate_model(model, test_loader):
	x_test, y_test = _loader_to_numpy(test_loader)
	x_test_t = torch.from_numpy(x_test.astype(np.float32))
	with torch.inference_mode():
		probs = model(x_test_t).detach().cpu().numpy().reshape(-1)
	predictions = (probs > 0.5).astype(np.float32)
	return float(np.mean(predictions == y_test))


def confusion_counts(model, test_loader):
	x_test, y_test = _loader_to_numpy(test_loader)
	x_test_t = torch.from_numpy(x_test.astype(np.float32))
	with torch.inference_mode():
		probs = model(x_test_t).detach().cpu().numpy().reshape(-1)
	predictions = (probs > 0.5).astype(np.int32)
	y_int = y_test.astype(np.int32)

	tn = int(np.sum((predictions == 0) & (y_int == 0)))
	fp = int(np.sum((predictions == 1) & (y_int == 0)))
	fn = int(np.sum((predictions == 0) & (y_int == 1)))
	tp = int(np.sum((predictions == 1) & (y_int == 1)))
	return tn, fp, fn, tp


def predict(model, x, y):
	input_tensor = torch.tensor([[x, y]], dtype=torch.float32)
	with torch.inference_mode():
		prob = float(model(input_tensor).item())
	pred = 1 if prob > 0.5 else 0
	confidence = prob if pred == 1 else (1.0 - prob)
	return pred, confidence


def save_cpn_weights(model, file_prefix="cpn_weights"):
	"""Persist Kohonen/Grossberg weights and CPN metadata to disk."""
	prefix_path = Path(file_prefix)
	npz_path = prefix_path.with_suffix(".npz")
	meta_path = prefix_path.with_suffix(".pkl")

	np.savez(
		npz_path,
		kohonen_weights=model.kohonen_weights.detach().cpu().numpy(),
		grossberg_weights=model.grossberg_weights.detach().cpu().numpy(),
	)
	metadata = {
		"n_kohonen": model.n_kohonen,
		"input_size": model.input_size,
		"output_size": model.output_size,
		"input_mode": model.input_mode,
	}
	with open(meta_path, "wb") as file_obj:
		pickle.dump(metadata, file_obj)

	return str(npz_path), str(meta_path)


def load_cpn_weights(file_prefix="cpn_weights"):
	"""Load a CPN model from saved weight files."""
	prefix_path = Path(file_prefix)
	npz_path = prefix_path.with_suffix(".npz")
	meta_path = prefix_path.with_suffix(".pkl")

	with open(meta_path, "rb") as file_obj:
		metadata = pickle.load(file_obj)

	arrays = np.load(npz_path)
	model = CPN(
		n_kohonen=int(metadata["n_kohonen"]),
		input_size=int(metadata["input_size"]),
		output_size=int(metadata["output_size"]),
		input_mode=str(metadata.get("input_mode", "augmented_unit_sphere")),
	)

	with torch.no_grad():
		model.kohonen_weights.copy_(torch.from_numpy(arrays["kohonen_weights"].astype(np.float32)))
		model.grossberg_weights.copy_(torch.from_numpy(arrays["grossberg_weights"].astype(np.float32)))

	return model


__all__ = [
	"CPN_ARCHITECTURES",
	"CPN_PARAMS",
	"CPN",
	"train_model",
	"evaluate_model",
	"confusion_counts",
	"predict",
	"save_cpn_weights",
	"load_cpn_weights",
]