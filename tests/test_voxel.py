import numpy as np
from dendromatics import voxel
from dendroptimized import voxelize


def test_voxel():
    xyz = np.random.rand(10_000_000, 3) * 100
    cloud_orig, vox2c_orig, _ = voxel.voxelate(xyz, 0.3, 0.3, 5, with_n_points=False)
    cloud_opti, vox2c_opti, _ = voxelize(xyz, 0.3, 0.3, 5)
    np.testing.assert_allclose(cloud_opti, cloud_orig)
    np.testing.assert_equal(vox2c_orig, vox2c_opti)

    # This test below won't work as neither dendromatics
    # nor dendroptimized use a stable sort
    # np.testing.assert_equal(c2vox_orig, c2vox_opti)
