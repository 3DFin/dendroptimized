import math

import numpy as np
import pytest
from sklearn.cluster import DBSCAN

from dendroptimized import connected_components, voxelize


def test_connected():
    rng = np.random.default_rng()
    xyz = rng.uniform(0, 100, size=(10_000_000, 3))
    cloud_opti, _, _ = voxelize(xyz, 0.3, 0.3, 5, with_n_points=False)
    labels = connected_components(cloud_opti, 0.3 * math.sqrt(3) + 1e-6, 2)

    dbscan = DBSCAN(eps=0.3 * math.sqrt(3) + 1e-6, min_samples=2, n_jobs=-1)
    dbscan.fit(cloud_opti)
    np.testing.assert_equal(np.unique(dbscan.labels_).shape, np.unique(labels).shape)
    np.testing.assert_equal(dbscan.labels_[dbscan.labels_ == -1], labels[labels == -1])


def fixture():
    rng = np.random.default_rng(seed=1)
    xyz = rng.uniform(0, 100, size=(5_000_000, 3))
    cloud_opti, _, _ = voxelize(xyz, 0.3, 0.3, 5, with_n_points=False)
    return cloud_opti

@pytest.mark.benchmark(group="Connected Components", disable_gc=True, warmup=True)
def test_bench_connected_dendromatics(benchmark):
    def _to_bench():
        DBSCAN(eps=0.3 * math.sqrt(3) + 1e-6, min_samples=2, n_jobs=-1).fit(fixture())

    benchmark(_to_bench)

@pytest.mark.benchmark(group="Connected Components", disable_gc=True, warmup=True)
def test_bench_connected_dendroptimized(benchmark):
    def _to_bench():
        connected_components(fixture(), 0.3 * math.sqrt(3) + 1e-6, 2)

    benchmark(_to_bench)
