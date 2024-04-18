
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/vector.h>

#include "voxel.hpp"

namespace nb = nanobind;
using namespace nb::literals;

NB_MODULE(dendroptimized, m)
{
    m.doc() = "Dendroptimized module, optimized Ad Hoc algorithms for dendromatic";
    m.def(
        "voxelate", &dendroptimized::voxelate_ll<double>, "xyz"_a.noconvert(), "res_xy"_a, "res_z"_a, "n_digits"_a,
        "id_x"_a = 0, "id_y"_a = 1, "id_z"_a = 2);
}
