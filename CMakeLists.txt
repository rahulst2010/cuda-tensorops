cmake_minimum_required(VERSION 3.10)
project(CUDA-TensorOps LANGUAGES CXX CUDA)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
find_package(CUDAToolkit REQUIRED)

# Shared library
add_library(cuda_tensor_ops SHARED
    src/gemm_kernel.cu
    src/layer_norm_kernel.cu
)
target_include_directories(cuda_tensor_ops PUBLIC include)
target_link_libraries(cuda_tensor_ops PRIVATE CUDA::cudart)
set_target_properties(cuda_tensor_ops PROPERTIES 
    CUDA_ARCHITECTURES "native"
    POSITION_INDEPENDENT_CODE ON
)
