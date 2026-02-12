import unittest
import numpy as np

from data import generate_intertwined_spirals, normalize_spirals
from MLPx6 import prepare_data


class TestDataUtilities(unittest.TestCase):
    def test_generate_all_returns_three_datasets(self):
        n = 200
        result = generate_intertwined_spirals(n=n, seed=7, plot=False, sampling_method="ALL")
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 3)
        rnd_data, ctr_data, edge_data = result
        self.assertEqual(len(rnd_data), 2 * n)
        self.assertEqual(len(ctr_data), 2 * n)
        self.assertEqual(len(edge_data), 2 * n)

    def test_generate_single_dataset_labels(self):
        n = 150
        data = generate_intertwined_spirals(n=n, seed=7, plot=False, sampling_method="RND")
        labels = {label for _, _, label in data}
        self.assertEqual(len(data), 2 * n)
        self.assertEqual(labels, {0, 1})

    def test_normalize_spirals_stats(self):
        data = generate_intertwined_spirals(n=200, seed=7, plot=False, sampling_method="RND")
        normalized, stats = normalize_spirals(data)
        self.assertEqual(len(normalized), len(data))
        mean = np.array(stats["mean"], dtype=float)
        std = np.array(stats["std"], dtype=float)
        self.assertEqual(mean.shape, (2,))
        self.assertEqual(std.shape, (2,))
        norm_xy = np.array([[x, y] for x, y, _ in normalized], dtype=float)
        self.assertTrue(np.allclose(norm_xy.mean(axis=0), np.zeros(2), atol=1e-6))
        self.assertTrue(np.allclose(norm_xy.std(axis=0), np.ones(2), atol=1e-6))

    def test_prepare_data_split_sizes(self):
        data = [(float(i), float(i + 1), i % 2) for i in range(100)]
        train_loader, val_loader, test_loader = prepare_data(
            data, test_split=0.2, val_split=0.2, batch_size=16
        )
        train_count = len(train_loader.dataset)
        val_count = len(val_loader.dataset)
        test_count = len(test_loader.dataset)
        self.assertEqual(test_count, 20)
        self.assertEqual(val_count, 16)
        self.assertEqual(train_count, 64)


if __name__ == "__main__":
    unittest.main()
