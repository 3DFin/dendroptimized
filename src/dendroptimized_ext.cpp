
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/variant.h>
#include <nanobind/stl/vector.h>

#include "voxel.hpp"

namespace nb = nanobind;
using namespace nb::literals;

NB_MODULE(dendroptimized_ext, m)
{
    m.doc() = "Dendroptimized module, optimized Ad Hoc algorithms for dendromatics";
    m.def(
        "voxelize", &dendroptimized::voxelize_stub<double>, "xyz"_a.noconvert(), "res_xy"_a, "res_z"_a,
        "n_digits"_a = 5, "id_x"_a = 0, "id_y"_a = 1, "id_z"_a = 2, "with_n_points"_a = true, "verbose"_a = true);
}
