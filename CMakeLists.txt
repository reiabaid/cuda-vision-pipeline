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

option(CVK_BUILD_TESTS "Build correctness tests (need OpenCV)" ON)

find_package(CUDAToolkit REQUIRED)
if(CVK_BUILD_TESTS)
  find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs)
endif()

add_library(cvk
  src/kernels/grayscale.cu
  src/kernels/blur.cu
  src/kernels/sobel.cu
  src/kernels/histogram.cu
  src/kernels/resize.cu
)
target_include_directories(cvk PUBLIC include)
target_link_libraries(cvk PUBLIC CUDA::cudart)
target_compile_options(cvk PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>)

foreach(name grayscale blur sobel histogram resize preprocess)
  add_executable(bench_${name} bench/bench_${name}.cu)
  target_include_directories(bench_${name} PRIVATE bench)
  target_link_libraries(bench_${name} PRIVATE cvk)
endforeach()

if(CVK_BUILD_TESTS)
  foreach(name grayscale blur sobel histogram resize preprocess)
    add_executable(test_${name} tests/test_${name}.cpp)
    target_include_directories(test_${name} PRIVATE tests)
    target_link_libraries(test_${name} PRIVATE cvk ${OpenCV_LIBS})
    add_test(NAME ${name} COMMAND test_${name})
  endforeach()

  add_executable(cvk_demo demo/demo.cpp)
  target_link_libraries(cvk_demo PRIVATE cvk ${OpenCV_LIBS})
endif()
