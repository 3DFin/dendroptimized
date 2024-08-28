#pragma once

#include <dset.h>

#include <nanoflann.hpp>

#include "types.hpp"

namespace nb = nanobind;

namespace dendroptimized
{

template <typename real_t>
static VecIndex<int32_t> connected_components(RefCloud<real_t> xyz, const real_t eps_radius, const uint32_t min_samples)
{
    using kd_tree_t = nanoflann::KDTreeEigenMatrixAdaptor<RefCloud<real_t>, 3, nanoflann::metric_L2_Simple>;

    kd_tree_t    kd_tree(3, xyz, 10);
    const real_t sq_search_radius = eps_radius * eps_radius;

    const Eigen::Index n_points = xyz.rows();

    tf::Executor                           executor;
    tf::Taskflow                           taskflow;
    std::vector<std::vector<Eigen::Index>> nn_cells;
    nn_cells.resize(n_points);
    std::vector<bool> is_core;
    is_core.resize(n_points);
    uint32_t          count_core = 0;
    VecIndex<int32_t> cluster_id(n_points);
    cluster_id.fill(-1);

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

            is_core[point_id] = num_found >= min_samples;  // we include the core sample itself
            std::vector<Eigen::Index> nn_ids;
            std::transform(
                result_set.begin(), result_set.end(), std::back_inserter(nn_ids),
                [&](const auto& result)
                {
                    if (result.first != point_id) return result.first;
                });
            nn_cells[point_id] = std::move(nn_ids);
        },
        tf::StaticPartitioner(0));
    executor.run(taskflow).get();

    for (const auto core_ok : is_core)
    {
        if (core_ok) ++count_core;
    }

    DisjointSets uf(n_points);
    // TODO: union find (parallel disjoint set)
    auto link_core = taskflow.for_each_index(
        size_t(0), size_t(n_points), size_t(1),
        [&](size_t curr_id)
        {
            if (!is_core[curr_id]) return;
            for (const auto nn_id : nn_cells[curr_id])
            {
                if (is_core[nn_id] && curr_id > nn_id && !uf.same(curr_id, nn_id)) { uf.unite(curr_id, nn_id); }
            }
        });

    auto label_core = taskflow.for_each_index(
        size_t(0), size_t(n_points), size_t(1),
        [&](size_t curr_id)
        {
            if (!is_core[curr_id]) return;
            cluster_id[curr_id] = uf.find(curr_id);
        });

    // label other nodes,
    auto label_border = taskflow.for_each_index(
        size_t(0), size_t(n_points), size_t(1),
        [&](size_t curr_id)
        {
            if (!is_core[curr_id])
            {
                for (const auto nn_id : nn_cells[curr_id])
                {
                    real_t min_dist = std::numeric_limits<real_t>::max();

                    if (is_core[nn_id])
                    {
                        real_t dist = (xyz.row(nn_id) - xyz.row(curr_id)).squaredNorm();
                        if (dist < min_dist)
                        {
                            min_dist            = dist;
                            cluster_id[curr_id] = cluster_id[nn_id];
                        }
                    }
                }
            }
        });
    label_core.succeed(link_core);
    label_border.succeed(label_core);

    executor.run(taskflow).get();

    return cluster_id;
    // TODO make this parallel (do not forget to move the executor)
}
}  // namespace dendroptimized
