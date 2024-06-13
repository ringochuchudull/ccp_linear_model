cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(inference)

set(Torch_DIR, "/mnt/c/Users/CHURI/Downloads/cpp/libtorch/share/cmake/Torch")
set(pybind11_DIR, "/home/ringocsw/anaconda3/envs/tsl/lib/python3.8/site-packages/pybind11/share/cmake/pybind11")

find_package(Torch REQUIRED)
find_package(pybind11 REQUIRED)


pybind11_add_module(inference SHARED inference.cpp)
target_link_libraries(inference PRIVATE ${TORCH_LIBRARIES})
set_property(TARGET inference PROPERTY CXX_STANDARD 11)