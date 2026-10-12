import unittest

import numpy as np

from partmm.matching import frame_match_distances


class MatchConfidenceTest(unittest.TestCase):
    def test_distance_keeps_absolute_quality_not_only_knn_rank(self):
        source = np.zeros((9, 2, 4))
        source[..., 0] = 1
        query = source.copy()
        query[:, 1] = [np.cos(0.5), 0, 0, np.sin(0.5)]
        distances = frame_match_distances(source, query, np.eye(2, dtype=bool), 7)
        np.testing.assert_allclose(distances[:, 0], 0, atol=1e-14)
        np.testing.assert_allclose(distances[:, 1], 1, atol=1e-14)

    def test_single_frame_and_short_sequences_are_covered(self):
        source = np.array([[[1., 0, 0, 0]]])
        distances = frame_match_distances(source, np.repeat(source, 3, axis=0), [[True]], 7)
        np.testing.assert_array_equal(distances, np.zeros((3, 1)))


if __name__ == '__main__':
    unittest.main()
