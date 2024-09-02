
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/variant.h>
#include <nanobind/stl/vector.h>

#include "connected_components.hpp"
#include "voxel.hpp"

namespace nb = nanobind;
using namespace nb::literals;

NB_MODULE(dendroptimized_ext, m)
{
    m.doc() = "Dendroptimized module, optimized Ad Hoc algorithms for Dendromatics";
    m.def(
        "voxelize", &dendroptimized::voxelize_wrapper<double>, "xyz"_a.noconvert(), "res_xy"_a, "res_z"_a,
        "n_digits"_a = 5, "id_x"_a = 0, "id_y"_a = 1, "id_z"_a = 2, "with_n_points"_a = true, "verbose"_a = true, R"(
        Function used to voxelate point clouds. It allows to use a different
        resolution for (z), but (x, y) will always share the same resolution. It
        also allows to revert the process, by creating a unique code for each point
        in the point cloud, thus voxelated cloud can be seamlessly reverted to the
        original point cloud.
        
        Parameters
        ----------
        xyz : numpy.ndarray
            The point cloud to be voxelated. It is expected to have X, Y, Z fields.
            3D or higher array containing data with `float` type.
        resolution_xy : float
            (x, y) voxel resolution.
        resolution_z : float
            (z) voxel resolution.
        n_digits : int
            no op this parameter is only meant to mimick original interface
        X_field : int
            Index at which (x) coordinate is stored. Defaults to 0.
        Y_field : int
            Index at which (y) coordinate is stored. Defaults to 1.
        Z_field : int
            Index at which (z) coordinate is stored. Defaults to 2.
        verbose : boolean
            Indicates whether logs should be displayed
        with_n_points : boolean
            If True, output voxelated cloud will have a field including the number
            of points that each voxel contains. Defaults to True.

        Returns
        -------
        voxelated_cloud : numpy.ndarray
            The voxelated cloud. It consists of 3 columns, each with (x), (y) and
            (z) coordinates, and an optional 4th column having the number of points
            included in each voxel if with_n_points = True.
        vox_to_cloud_ind : numpy.ndarray
            Vector containing the indexes to revert to the original point cloud
            from the voxelated cloud.
        cloud_to_vox_ind : numpy.ndarray
            Vector containing the indexes to directly go from the original point
            cloud to the voxelated cloud.
        )");
    m.def(
        "connected_components", &dendroptimized::connected_components<double>, "xyz"_a.noconvert(), "eps"_a, "min_samples"_a = 2, R"(
        Simplified DBSCAN implementation.
        The cloud is intended to be a voxelized point cloud with isotropic resolution across all three dimensions. 
        The eps parameter should be set to the radius of a sphere enclosing a voxel (i.e., voxel side length x sqrt(3)). 
        Neighborhoods are initially computed in parallel using a KD-tree acceleration structure. 
        Core points (with a neighborhood size >= min_samples) are then connected using an efficient Union-Find algorithm. 
        Finally, border points are linked to their nearest cluster.

        like sklearn dbscan implementation, the query point itself is part of the set of samples. 
        the overall implementation should be faster and use less memory than sklearn alternative

        Parameters
        ----------
        xyz : numpy.ndarray
            The point cloud to be voxelated. It is expected to have X, Y, Z fields.
            3D or higher array containing data with `float` type.
        eps : float
            radius of the sphere used to define de neighborhood.
        min_samples : int
            minimum number of points in a point neighborhood required for it to be considered a core point.

        Returns
        -------
        cluster_labels : numpy.ndarray
            A vector with a length equal to the number of points, where each element contains 
            the ID of the cluster the point belongs to. 
            A value of ‘-1’ indicates that the point is considered to be noise.
        )");
}
