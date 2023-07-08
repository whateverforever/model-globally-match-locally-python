project(ppf_accelerator)
cmake_minimum_required(VERSION 3.19)

find_package (Eigen3 3.3 REQUIRED NO_MODULE)
find_package(Python 3.10 EXACT COMPONENTS Interpreter Development.Module REQUIRED)
include_directories(${PYTHON_INCLUDE_DIRS})

add_subdirectory(nanobind)
nanobind_add_module(ppf_fast ppf.cc)

target_link_libraries(ppf_fast PRIVATE ${PYTHON_LIBRARIES})
target_link_libraries(ppf_fast PUBLIC Eigen3::Eigen)
target_compile_features(ppf_fast PRIVATE cxx_std_17)
