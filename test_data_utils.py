import unittest
from typing import Sized, cast

from data import generate_interlocked_region_data
from MLPx6 import prepare_data


class TestDataUtilities(unittest.TestCase):
    @staticmethod
    def _dataset_size(loader):
        dataset = loader.dataset
        if hasattr(dataset, "tensors"):
            return int(dataset.tensors[0].shape[0])
        return len(cast(Sized, dataset))

    def test_generate_all_returns_three_datasets(self):
        n = 200
        result = generate_interlocked_region_data(n=n, seed=7, plot=False, sampling_method="ALL")
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 3)
        rnd_data, ctr_data, edge_data = result
        self.assertEqual(len(rnd_data), 2 * n)
        self.assertEqual(len(ctr_data), 2 * n)
        self.assertEqual(len(edge_data), 2 * n)

    def test_generate_single_dataset_labels(self):
        n = 150
        data = generate_interlocked_region_data(n=n, seed=7, plot=False, sampling_method="RND")
        labels = {label for _, _, label in data}
        self.assertEqual(len(data), 2 * n)
        self.assertEqual(labels, {0, 1})

    def test_prepare_data_split_sizes(self):
        data = [(float(i), float(i + 1), i % 2) for i in range(100)]
        train_loader, val_loader, test_loader = prepare_data(
            data, test_split=0.2, val_split=0.2, batch_size=16
        )
        train_count = self._dataset_size(train_loader)
        val_count = self._dataset_size(val_loader)
        test_count = self._dataset_size(test_loader)
        self.assertEqual(test_count, 20)
        self.assertEqual(val_count, 16)
        self.assertEqual(train_count, 64)


if __name__ == "__main__":
    unittest.main()
