cmake_minimum_required(VERSION 3.15...3.27)
project(dendroptimized)
find_package(Python 3.8 COMPONENTS Interpreter Development.Module REQUIRED)

if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

# Detect the installed nanobind package and import it into CMake
execute_process(
  COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
  OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE NB_DIR)
list(APPEND CMAKE_PREFIX_PATH "${NB_DIR}")

find_package(nanobind CONFIG REQUIRED)
find_package(Threads REQUIRED)

if(MSVC)
  # Static link MSVC rt for maximum compatibility
  # It avoid to mess with embedded MSVC rt from other packages (see PyQT5)
  set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
  # Same, enabling LTO (LTGC for MS) give less ABI compatibility, so we dissable it.
  # see https://learn.microsoft.com/en-us/cpp/porting/binary-compat-2015-2017?view=msvc-170
  # for more details.
  nanobind_add_module(dendroptimized_ext NOMINSIZE STABLE_ABI src/dendroptimized_ext.cpp)
else()
  # we enable LTO on Linux and MacOS.
  nanobind_add_module(dendroptimized_ext NOMINSIZE LTO STABLE_ABI src/dendroptimized_ext.cpp)
endif()

target_link_libraries(dendroptimized_ext PRIVATE Threads::Threads)

# All libs are header only.
# it's faster to include like this than using exported targets
# (i.e add_subdirectories(...))
target_include_directories(dendroptimized_ext PRIVATE "include"
  "third_party/eigen"
  "third_party/taskflow"
  "third_party/nanoflann/include"
  "third_party/dset")

install(TARGETS dendroptimized_ext LIBRARY DESTINATION dendroptimized)
