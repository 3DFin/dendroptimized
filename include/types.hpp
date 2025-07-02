#pragma once

#include <nanobind/eigen/dense.h>

#include <Eigen/Dense>
#include <Eigen/Eigenvalues>

namespace nb = nanobind;

namespace dendroptimized
{

template <typename real_t>
using PointCloud3 = Eigen::Matrix<real_t, Eigen::Dynamic, 3, Eigen::RowMajor>;

template <typename real_t>
using PointCloud2 = Eigen::Matrix<real_t, Eigen::Dynamic, 2, Eigen::RowMajor>;

template <typename real_t>
using RefCloud3 = Eigen::Ref<const PointCloud3<real_t>>;

template <typename real_t>
using RefCloud2 = Eigen::Ref<const PointCloud2<real_t>>;

template <typename real_t>
using PointCloudAugmented = Eigen::Matrix<real_t, Eigen::Dynamic, 4, Eigen::RowMajor>;

template <typename real_t>
using Vec3 = Eigen::RowVector<real_t, 3>;

template <typename int_t>
using VecIndex = Eigen::RowVector<int_t, Eigen::Dynamic>;

template <typename real_t>
using MatrixCloud = Eigen::Matrix<real_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
template <typename real_t>
using DRefMatrixCloud = nb::DRef<const MatrixCloud<real_t>>;

}  // namespace dendroptimized
