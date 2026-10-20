cmake_minimum_required(VERSION 3.24)
# Default: RTX 3050 Ti Laptop = compute capability 8.6. Override with
# -DCMAKE_CUDA_ARCHITECTURES=... This must come BEFORE project(): once CUDA is enabled,
# CMake has already filled in its own default (sm_52 here) and the check would never fire.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 86 CACHE STRING "CUDA architectures")
endif()

project(cuda_vision_pipeline LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
enable_testing()

find_package(CUDAToolkit REQUIRED)
find_package(OpenCV REQUIRED COMPONENTS core imgproc)

add_library(cvk
  src/kernels/grayscale.cu
)
target_include_directories(cvk PUBLIC include)
target_link_libraries(cvk PUBLIC CUDA::cudart)
target_compile_options(cvk PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>)

add_executable(bench_grayscale bench/bench_grayscale.cu)
target_include_directories(bench_grayscale PRIVATE bench)
target_link_libraries(bench_grayscale PRIVATE cvk)

add_executable(test_grayscale tests/test_grayscale.cpp)
target_link_libraries(test_grayscale PRIVATE cvk ${OpenCV_LIBS})
add_test(NAME grayscale COMMAND test_grayscale)
