#pragma once

#include <nanoflann.hpp>

#include "types.hpp"

namespace nb = nanobind;

namespace dendroptimized
{

template <typename real_t>
static void connected_components(RefCloud<real_t> xyz, const real_t eps_radius)
{
    using kd_tree_t = nanoflann::KDTreeEigenMatrixAdaptor<RefCloud<real_t>, 3, nanoflann::metric_L2_Simple>;

    kd_tree_t    kd_tree(3, xyz, 10);
    const real_t sq_search_radius = eps_radius * eps_radius;

    const Eigen::Index n_points = xyz.rows();
    std::cout << n_points << std::endl;
    std::cout << eps_radius << std::endl;

    tf::Executor                           executor;
    tf::Taskflow                           taskflow;
    std::vector<std::vector<Eigen::Index>> nn_cells;
    nn_cells.resize(n_points);

    taskflow.for_each_index(
        Eigen::Index(0), n_points, Eigen::Index(1),
        [&](Eigen::Index point_id)
        {
            std::vector<nanoflann::ResultItem<Eigen::Index, real_t>> result_set;

            nanoflann::RadiusResultSet<real_t, Eigen::Index> radius_result_set(sq_search_radius, result_set);
            const auto                                       num_found =
                kd_tree.index_->radiusSearchCustomCallback(xyz.row(point_id).data(), radius_result_set);

            if (num_found > 28)
            {
                // TODO: throw as it's too much, we only expect 27 NN + the base point
                throw std::invalid_argument(
                    "it seems that your radius is too big, CC extraction is only meant to be used in "
                    "27-connectivity "
                    "context on regular voxels grids");
            }
            std::vector<Eigen::Index> ids;
            std::transform(
                result_set.begin(), result_set.end(), std::back_inserter(ids),
                [&](const auto& result) { if(result.second != point_id) return result.second; });
            nn_cells[point_id] = std::move(ids);
        },
        tf::StaticPartitioner(0));
    
    // TODO: union find (parallel disjoint set)

    // TODO: prune node with cluster < parameter size (to add in the signature)

    // TODO : label other nodes.
    // in our config border points should not exists, we should only have core points and noise points
    executor.run(taskflow).get();
}
}  // namespace dendroptimized
