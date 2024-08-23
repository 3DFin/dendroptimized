import numpy as np
from dendromatics import voxel
from dendroptimized import voxelize, connected_components

def vox():
#def test_voxel():
    xyz = np.random.rand(10_000_000, 3) * 100
    cloud_orig, vox2c_orig, _ = voxel.voxelate(xyz, 0.3, 0.3, 5, with_n_points=True)
    cloud_opti, vox2c_opti, _ = voxelize(xyz, 0.3, 0.3, 5, with_n_points=True)
    np.testing.assert_allclose(cloud_opti, cloud_orig)
    np.testing.assert_equal(vox2c_orig, vox2c_opti)

    # This test below won't work as neither dendromatics
    # nor dendroptimized use a stable sort
    # np.testing.assert_equal(c2vox_orig, c2vox_opti)


def test_connected():
    xyz = np.random.rand(10_000_000, 3) * 100
    cloud_opti, _, _ = voxelize(xyz, 0.3, 0.3, 5, with_n_points=False)
    connected_components(cloud_opti, 0.3 * 1.9)
    
def fixture():
    np.random.seed(1)
    return np.random.uniform(0, 100, size=(10_000_000, 3))


def test_bench_dendromatics(benchmark):
    def _to_bench():
        voxel.voxelate(fixture(), 0.3, 0.3, 5, with_n_points=False)

    benchmark(_to_bench)


def test_bench_dendroptimized(benchmark):
    def _to_bench():
        voxelize(fixture(), 0.3, 0.3, 5)

    benchmark(_to_bench)
