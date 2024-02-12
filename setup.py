from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension
import platform
import os

# targets and compile options
name = "dendroptimized"
include_dirs = ["include", "taskflow/", "eigen"]

# user can define a path to their own Eigen lib
DENDROPTIMIZED_EIGEN_LIB_PATH = os.environ.get("DENDROPTIMIZED_EIGEN_LIB_PATH", None)

if DENDROPTIMIZED_EIGEN_LIB_PATH is not None:
    include_dirs.pop()  # pop current eigen dir
    include_dirs.append(DENDROPTIMIZED_EIGEN_LIB_PATH)

# Compilation and linkage options
if platform.system() == "Windows":
    extra_compile_args = ["/std:c++17"]
    extra_link_args = []
elif platform.system() == "Linux":
    extra_compile_args = ["-std=c++17", "-pthread"]
    extra_link_args = []
elif platform.system() == "Darwin":
    extra_compile_args = ["-std=c++17"]
    extra_link_args = []
else:
    raise NotImplementedError("OS not yet supported.")

#  Compilation
pybindmodule = Pybind11Extension(
    name,
    # list source files
    ["./wrapper/dendroptimized_cpy.cpp"],
    include_dirs=include_dirs,
    extra_compile_args=extra_compile_args,
    extra_link_args=extra_link_args,
)

setup(name=name, package_dir={"": "wrapper"}, ext_modules=[pybindmodule])
