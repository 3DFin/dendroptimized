import numpy as np
import pytest

from dendromatics import voxel
from dendroptimized import voxelize


def test_voxel():
    rng = np.random.default_rng()
    xyz = rng.uniform(0, 100, size=(1_000_000, 3))
    cloud_orig, vox2c_orig, _ = voxel.voxelate(xyz, 0.3, 0.3, 5, with_n_points=True)
    cloud_opti, vox2c_opti, _ = voxelize(xyz, 0.3, 0.3, 5, with_n_points=True)
    np.testing.assert_allclose(cloud_opti, cloud_orig)
    np.testing.assert_equal(vox2c_orig, vox2c_opti)

    # This test below won't work as neither dendromatics
    # nor dendroptimized use a stable sort
    # np.testing.assert_equal(c2vox_orig, c2vox_opti)


def fixture():
    rng =  np.random.default_rng()
    return rng.uniform(0, 100, size=(1_000_000, 3))

@pytest.mark.benchmark(group="Voxel", disable_gc=True, warmup=True)
def test_bench_voxel_dendromatics(benchmark):
    def _to_bench():
        voxel.voxelate(fixture(), 0.3, 0.3, 5, with_n_points=False)

    benchmark(_to_bench)

@pytest.mark.benchmark(group="Voxel", disable_gc=True, warmup=True)
def test_bench_voxel_dendroptimized(benchmark):
    def _to_bench():
        voxelize(fixture(), 0.3, 0.3, 5)

    benchmark(_to_bench)
