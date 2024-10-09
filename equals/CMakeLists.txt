cmake_minimum_required(VERSION 3.17.0)
set(CMAKE_GENERATOR "Ninja")
set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc")
set(CMAKE_CUDA_ARCHITECTURES 86)

project(bench_cubfindif_backend VERSION 0.1.0 LANGUAGES CUDA CXX)

add_executable(bench_equal bench_equal.cu)
# target_compile_options(playground PRIVATE "--extended-lambda") 

target_include_directories(bench_equal PRIVATE "/home/ggonidelis/cccl/thrust")
target_include_directories(bench_equal PRIVATE "/home/ggonidelis/cccl/cub")
target_include_directories(bench_equal PRIVATE "/home/ggonidelis/cccl/libcudacxx/include")

add_executable(bench_find bench_find.cu)

target_include_directories(bench_find PRIVATE "/home/ggonidelis/cccl/thrust")
target_include_directories(bench_find PRIVATE "/home/ggonidelis/cccl/cub")
target_include_directories(bench_find PRIVATE "/home/ggonidelis/cccl/libcudacxx/include")