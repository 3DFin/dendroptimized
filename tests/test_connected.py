import numpy as np
import math
import pytest
from dendroptimized import voxelize, connected_components
from sklearn.cluster import DBSCAN

def test_connected():
    xyz = np.random.rand(1_000_000, 3) * 100
    cloud_opti, _, _ = voxelize(xyz, 0.3, 0.3, 5, with_n_points=False)
    labels = connected_components(cloud_opti, 0.3 * math.sqrt(3) + 1e-6, 2)

    dbscan = DBSCAN(eps=0.3 * math.sqrt(3) + 1e-6, min_samples=2, n_jobs=-1)
    dbscan.fit(cloud_opti)
    np.testing.assert_equal(np.unique(dbscan.labels_).shape, np.unique(labels).shape)
    np.testing.assert_equal(dbscan.labels_[dbscan.labels_ == -1], labels[labels == -1])


def fixture():
    np.random.seed(1)
    xyz = np.random.uniform(0, 100, size=(5_000_000, 3))
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
