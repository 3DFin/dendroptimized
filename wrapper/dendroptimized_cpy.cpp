
#include <pybind11/iostream.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "voxel.hpp"

namespace py = pybind11;
using namespace pybind11::literals;

PYBIND11_MODULE(dendroptimized, m)
{
    m.doc() = "Dendroptimized module";
    m.def(
        "voxelate", &dendroptimized::voxelate_ll<double>, "xyz"_a.noconvert(), "res_xy"_a, "res_z"_a, "n_digits"_a,
        "id_x"_a = 0, "id_y"_a = 1, "id_z"_a = 2);
    //m.def(
    //    "voxelate_legacy", &dendroptimized::voxelate<double>, "xyz"_a.noconvert(), "res_xy"_a, "res_z"_a, "n_digits"_a,
    //    "id_x"_a = 0, "id_y"_a = 1, "id_z"_a = 2);
}
