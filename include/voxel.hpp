
#pragma once

#include <Eigen/Dense>
#include <chrono>
#include <iostream>
#include <taskflow/algorithm/for_each.hpp>
#include <taskflow/algorithm/scan.hpp>
#include <taskflow/algorithm/sort.hpp>
#include <taskflow/algorithm/transform.hpp>

#include "types.hpp"

namespace py = pybind11;

namespace dendroptimized
{

// baseline version, using C++ hash map
template <typename real_t>
static std::tuple<py::array_t<real_t>, py::array_t<uint64_t>, py::array_t<uint64_t>> voxelate(
    DRefMatrixCloud<real_t> xyz, const real_t res_xy, const real_t res_z, const uint32_t n_digits, const uint32_t id_x,
    const uint32_t id_y, const uint32_t id_z)
{
    // The coordinate minima
    const auto               start      = std::chrono::high_resolution_clock::now();
    const Vec3<real_t>       min_vec    = xyz.colwise().minCoeff();
    const PointCloud<real_t> shifted_pc = xyz.rowwise() - min_vec;

    std::map<uint64_t, std::vector<uint64_t>> voxels;

    const uint64_t z_shift = std::pow(10, n_digits * 2);
    const uint64_t y_shift = std::pow(10, n_digits);

    // function for voxel hashing
    const auto create_hash = [&](const Vec3<real_t>& point) -> uint64_t
    {
        return static_cast<uint64_t>(std::floor((point(id_z) / res_z))) * z_shift +
               static_cast<uint64_t>(std::floor((point(id_y) / res_xy))) * y_shift +
               static_cast<uint64_t>(std::floor((point(id_x) / res_xy)));
    };

    const auto start_hashing = std::chrono::high_resolution_clock::now();

    for (Eigen::Index point_id = 0; point_id < shifted_pc.rows(); ++point_id)
    {
        const auto hash = create_hash(shifted_pc.row(point_id));
        voxels[hash].push_back(point_id);
    }

    std::cout << "Hashing Time "
              << std::chrono::duration_cast<std::chrono::milliseconds>(
                     std::chrono::high_resolution_clock::now() - start_hashing)
                     .count()
              << "ms" << std::endl;

    const auto start_vox = std::chrono::high_resolution_clock::now();
    uint64_t   voxel_id  = 0;

    PointCloud<real_t> vox_pc(voxels.size(), 3);
    VecIndex<uint64_t> cloud_to_vox_ind(shifted_pc.rows());
    VecIndex<uint64_t> vox_to_cloud_ind(voxels.size());

    const auto z_code = [&](const uint64_t unique_code) { return unique_code / z_shift; };
    const auto y_code = [&](const uint64_t unique_code, const uint64_t z_code)
    { return (unique_code - z_code * z_shift) / y_shift; };
    const auto x_code = [&](const uint64_t unique_code, const uint64_t z_code, const uint64_t y_code)
    { return unique_code - z_code * z_shift - y_code * y_shift; };

    const real_t centroid_shift_x = min_vec(0) + res_xy / real_t(2.0);
    const real_t centroid_shift_y = min_vec(1) + res_xy / real_t(2.0);
    const real_t centroid_shift_z = min_vec(2) + res_z / real_t(2.0);

    std::map<uint64_t, uint64_t> voxels_id_contiguous;

    std::for_each(
        voxels.cbegin(), voxels.cend(),
        [&](auto& it)
        {
            const auto voxel_hash            = it.first;
            voxels_id_contiguous[voxel_hash] = voxel_id;
            ++voxel_id;
        });

    tf::Executor executor;
    tf::Taskflow taskflow;
    taskflow.for_each(
        voxels.cbegin(), voxels.cend(),
        [&](auto& voxel_it)
        {
            const auto     voxel_hash = voxel_it.first;
            const auto&    indices    = voxel_it.second;
            const uint64_t voxel_id   = voxels_id_contiguous[voxel_hash];

            const uint64_t z_code_val = z_code(voxel_hash);
            const uint64_t y_code_val = y_code(voxel_hash, z_code_val);
            const uint64_t x_code_val = x_code(voxel_hash, z_code_val, y_code_val);
            vox_pc.row(voxel_id)      = Vec3<real_t>(
                x_code_val * res_xy + centroid_shift_x, y_code_val * res_xy + centroid_shift_y,
                z_code_val * res_z + centroid_shift_z);

            for (const auto& index : indices) { cloud_to_vox_ind(index) = voxel_id; }
            vox_to_cloud_ind(voxel_id) = indices[0];
        });

    executor.run(taskflow).get();

    std::cout << "Voxelization Creation Time "
              << std::chrono::duration_cast<std::chrono::milliseconds>(
                     std::chrono::high_resolution_clock::now() - start_vox)
                     .count()
              << "ms" << std::endl;

    std::cout << "Total Voxelization Time "
              << std::chrono::duration_cast<std::chrono::milliseconds>(
                     std::chrono::high_resolution_clock::now() - start)
                     .count()
              << "ms" << std::endl;

    std::cout << "Number of voxels " << voxels.size() << std::endl;

    std::cout << "Voxels account for " << vox_pc.rows() * 100 / vox_to_cloud_ind.size() << "% of original points"
              << std::endl;

    return std::make_tuple(
        py::array_t<real_t>(std::vector<ptrdiff_t>{static_cast<py::ssize_t>(vox_pc.rows()), 3}, vox_pc.data()),
        py::array_t<uint64_t>(
            std::vector<ptrdiff_t>{static_cast<py::ssize_t>(cloud_to_vox_ind.size())}, cloud_to_vox_ind.data()),
        py::array_t<uint64_t>(
            std::vector<ptrdiff_t>{static_cast<py::ssize_t>(vox_to_cloud_ind.size())}, vox_to_cloud_ind.data()));
}

// low level version with Taskflow
template <typename real_t>
static std::tuple<py::array_t<real_t>, py::array_t<uint64_t>, py::array_t<uint64_t>> voxelate_ll(
    DRefMatrixCloud<real_t> xyz, const real_t res_xy, const real_t res_z, const uint32_t n_digits, const uint32_t id_x,
    const uint32_t id_y, const uint32_t id_z)
{
    // The coordinate minima
    const auto start_total = std::chrono::high_resolution_clock::now();

    std::cout << "-Voxelization " << std::endl;
    std::cout << "  Voxel resolution " << res_xy << " x " << res_xy << " x " << res_z << std::endl;

    tf::Executor executor;
    tf::Taskflow tf;

    const Eigen::Index num_points = xyz.rows();

    // parallel min coeff + sorted indices initialization
    real_t min_x, min_y, min_z;
    tf.emplace(
        [&]()
        {
            min_x = xyz(0, id_x);
            for (Eigen::Index point_id = 1; point_id < num_points; ++point_id)
            {
                if (xyz(point_id, id_x) < min_x) min_x = xyz(point_id, id_x);
            };
        });
    tf.emplace(
        [&]()
        {
            min_y = xyz(0, id_y);
            for (Eigen::Index point_id = 1; point_id < num_points; ++point_id)
            {
                if (xyz(point_id, id_y) < min_y) min_y = xyz(point_id, id_y);
            };
        });
    tf.emplace(
        [&]()
        {
            min_z = xyz(0, id_z);
            for (Eigen::Index point_id = 1; point_id < num_points; ++point_id)
            {
                if (xyz(point_id, id_z) < min_z) min_z = xyz(point_id, id_z);
            };
        });

    executor.run(tf).wait();

    const Vec3<real_t> min_vec(min_x, min_y, min_z);

    const uint64_t z_shift = std::pow(10, n_digits * 2);
    const uint64_t y_shift = std::pow(10, n_digits);

    // function for voxel hashing
    const auto create_hash = [&](const Vec3<real_t>& point) -> uint64_t
    {
        return static_cast<uint64_t>(std::floor((point(id_z) - min_vec(id_z)) / res_z)) * z_shift +
               static_cast<uint64_t>(std::floor((point(id_y) - min_vec(id_y)) / res_xy)) * y_shift +
               static_cast<uint64_t>(std::floor((point(id_x) - min_vec(id_x)) / res_xy));
    };

    const auto z_code = [&](const uint64_t unique_code) { return unique_code / z_shift; };
    const auto y_code = [&](const uint64_t unique_code, const uint64_t z_code)
    { return (unique_code - z_code * z_shift) / y_shift; };
    const auto x_code = [&](const uint64_t unique_code, const uint64_t z_code, const uint64_t y_code)
    { return unique_code - z_code * z_shift - y_code * y_shift; };

    const real_t centroid_shift_x = min_vec(0) + res_xy / real_t(2.0);
    const real_t centroid_shift_y = min_vec(1) + res_xy / real_t(2.0);
    const real_t centroid_shift_z = min_vec(2) + res_z / real_t(2.0);

    // first create an index vector

    std::vector<uint64_t> hashes(num_points);
    VecIndex<uint32_t>    cloud_to_vox_ind(num_points);
    PointCloud<real_t>    vox_pc;
    VecIndex<uint32_t>    vox_to_cloud_ind;

    std::vector<uint32_t> first_point_in_vox(num_points, 0);
    first_point_in_vox[0] = 1;

    std::chrono::time_point<std::chrono::high_resolution_clock> start_hashing, stop_hashing, start_sorting,
        stop_sorting, start_grouping, stop_grouping, start_voxelization, stop_voxelization;

    std::vector<Eigen::Index>           sorted_indices(num_points);
    std::vector<Eigen::Index>::iterator first_it = sorted_indices.begin();
    std::vector<Eigen::Index>::iterator last_it  = sorted_indices.end();
    std::iota(first_it, last_it, 0);

    auto start_hashing_task =
        tf.emplace([&start_hashing]() { start_hashing = std::chrono::high_resolution_clock::now(); });
    auto stop_hashing_task =
        tf.emplace([&stop_hashing]() { stop_hashing = std::chrono::high_resolution_clock::now(); });

    auto start_sorting_task =
        tf.emplace([&start_sorting]() { start_sorting = std::chrono::high_resolution_clock::now(); });
    auto stop_sorting_task =
        tf.emplace([&stop_sorting]() { stop_sorting = std::chrono::high_resolution_clock::now(); });

    auto start_grouping_task =
        tf.emplace([&start_grouping]() { start_grouping = std::chrono::high_resolution_clock::now(); });
    auto stop_grouping_task =
        tf.emplace([&stop_grouping]() { stop_grouping = std::chrono::high_resolution_clock::now(); });

    auto start_voxelization_task =
        tf.emplace([&start_voxelization]() { start_voxelization = std::chrono::high_resolution_clock::now(); });
    auto stop_voxelization_task =
        tf.emplace([&stop_voxelization]() { stop_voxelization = std::chrono::high_resolution_clock::now(); });

    // create hashes
    auto hashing = tf.for_each(
                         std::ref(first_it), std::ref(last_it),
                         [&](const Eigen::Index point_id) { hashes[point_id] = create_hash(xyz.row(point_id)); })
                       .name("hashing");
    ;

    // second order point by dimensions
    auto sort_indices = tf.sort(
                              std::ref(first_it), std::ref(last_it),
                              [&](const Eigen::Index a, const Eigen::Index b) { return hashes[a] < hashes[b]; })
                            .name("sort_indices");

    // unique
    auto unique = tf.for_each_index(
                        Eigen::Index(1), Eigen::Index(num_points), Eigen::Index(1),
                        [&](const Eigen::Index point_id)
                        {
                            if (hashes[sorted_indices[point_id]] != hashes[sorted_indices[point_id - 1]])
                            {
                                first_point_in_vox[point_id] = 1;
                            }
                        })
                      .name("unique");

    // count and generate voxel ids
    auto count_voxels =
        tf.inclusive_scan(
              first_point_in_vox.begin(), first_point_in_vox.end(), first_point_in_vox.begin(), std::plus<int>{})
            .name("count_voxels");

    // allocate voxel point cloud and vox_to_cloud
    auto allocate = tf.emplace(
        [&]()
        {
            vox_pc           = PointCloud<real_t>(first_point_in_vox.back(), 3);
            vox_to_cloud_ind = VecIndex<uint32_t>(first_point_in_vox.back());
        });

    auto fill_vox_pc = tf.for_each_index(
                             Eigen::Index(0), Eigen::Index(num_points), Eigen::Index(1),
                             [&](const Eigen::Index point_id)
                             {
                                 const auto voxel_id             = first_point_in_vox[point_id] - 1;  // it starts at 1
                                 const auto real_point_id        = sorted_indices[point_id];
                                 cloud_to_vox_ind(real_point_id) = voxel_id;
                                 //  we account for the first point here
                                 //  maybe it could be better to init. it in the allocation tasks
                                 if (point_id == 0 || voxel_id != first_point_in_vox[point_id - 1] - 1)
                                 {
                                     const uint64_t hash_val = hashes[real_point_id];

                                     const uint64_t z_code_val = z_code(hash_val);
                                     const uint64_t y_code_val = y_code(hash_val, z_code_val);
                                     const uint64_t x_code_val = x_code(hash_val, z_code_val, y_code_val);

                                     vox_pc.row(voxel_id) = Vec3<real_t>(
                                         x_code_val * res_xy + centroid_shift_x, y_code_val * res_xy + centroid_shift_y,
                                         z_code_val * res_z + centroid_shift_z);
                                     vox_to_cloud_ind(voxel_id) = real_point_id;
                                 }
                             })
                           .name("fill_vox_pc");

    // Taskflow workflow
    // Group timings task here
    start_hashing_task.precede(hashing);
    stop_hashing_task.succeed(hashing);
    start_sorting_task.precede(sort_indices);
    stop_sorting_task.succeed(sort_indices);
    start_grouping_task.precede(unique);
    stop_grouping_task.succeed(count_voxels);
    start_voxelization_task.precede(allocate);
    stop_voxelization_task.succeed(fill_vox_pc);

    // Group tasks here
    hashing.precede(sort_indices);
    sort_indices.precede(unique);
    unique.precede(count_voxels);
    count_voxels.precede(allocate);
    allocate.precede(fill_vox_pc);

    // Launch tasks
    executor.run(tf).wait();

    // output
    std::cout << "  Hashing Time "
              << std::chrono::duration_cast<std::chrono::milliseconds>(stop_hashing - start_hashing).count() << "ms"
              << std::endl;

    std::cout << "  Sorting Time "
              << std::chrono::duration_cast<std::chrono::milliseconds>(stop_sorting - start_sorting).count() << "ms"
              << std::endl;

    std::cout << "  Grouping Time "
              << std::chrono::duration_cast<std::chrono::milliseconds>(stop_grouping - start_grouping).count() << "ms"
              << std::endl;

    std::cout << "  Voxelization Time "
              << std::chrono::duration_cast<std::chrono::milliseconds>(stop_voxelization - start_voxelization).count()
              << "ms" << std::endl;

    std::cout << "  Total Time "
              << std::chrono::duration_cast<std::chrono::milliseconds>(
                     std::chrono::high_resolution_clock::now() - start_total)
                     .count()
              << "ms" << std::endl;

    std::cout << "  Number of voxels " << vox_pc.rows() << std::endl;

    std::cout << "  Voxels account for " << vox_pc.rows() * 100 / static_cast<double>(num_points)
              << "% of original points" << std::endl;

    return std::make_tuple(
        py::array_t<real_t>(std::vector<ptrdiff_t>{static_cast<py::ssize_t>(vox_pc.rows()), 3}, vox_pc.data()),
        py::array_t<uint32_t>(
            std::vector<ptrdiff_t>{static_cast<py::ssize_t>(cloud_to_vox_ind.size())}, cloud_to_vox_ind.data()),
        py::array_t<uint32_t>(
            std::vector<ptrdiff_t>{static_cast<py::ssize_t>(vox_to_cloud_ind.size())}, vox_to_cloud_ind.data()));
}

}  // namespace dendroptimized