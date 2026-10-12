import unittest

import numpy as np

from experiments.monocular_gaussian.raw_node_skinning import (
    compute_raw_node_weights,
)


class RawNodeSkinningTest(unittest.TestCase):
    def test_weights_use_hyperfeatures_and_normalize(self):
        points = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=np.float32)
        features = np.array([[0.0], [1.0]], dtype=np.float32)
        nodes = np.array(
            [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
            dtype=np.float32,
        )
        indices, weights = compute_raw_node_weights(
            points,
            features,
            nodes,
            node_radius=np.ones(2, dtype=np.float32),
            node_weight=np.ones(2, dtype=np.float32),
            k=2,
            chunk_size=1,
        )
        np.testing.assert_array_equal(indices[:, 0], [0, 1])
        np.testing.assert_allclose(weights.sum(axis=1), 1.0, atol=1e-6)
        self.assertGreater(weights[0, 0], weights[0, 1])
        self.assertGreater(weights[1, 0], weights[1, 1])


if __name__ == "__main__":
    unittest.main()
