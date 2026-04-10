cmake_minimum_required(VERSION 3.18)

project(dmk_benchmark LANGUAGES CXX)

find_package(OpenMP REQUIRED)

set(FIDMK_ROOT "/mnt/home/xgao1/codes/FIDMK" CACHE PATH "Path to the FIDMK checkout")
set(DMK_FMM3D_DIR "/mnt/home/xgao1/codes/FMM3D" CACHE PATH "Path to the local FMM3D checkout")
set(CMAKE_INSTALL_PREFIX "${CMAKE_BINARY_DIR}/install" CACHE PATH "" FORCE)
set(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT FALSE)

set(DMK_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(DMK_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(DMK_BUILD_PVFMM OFF CACHE BOOL "" FORCE)
set(DMK_BUILD_REFERENCE OFF CACHE BOOL "" FORCE)

add_subdirectory("${FIDMK_ROOT}" fidmk)

find_library(DMK_FMM3D_LIBRARY NAMES fmm3d
  PATHS "${DMK_FMM3D_DIR}/lib" "${DMK_FMM3D_DIR}/lib-static"
  NO_DEFAULT_PATH
)

if (NOT DMK_FMM3D_LIBRARY)
  message(FATAL_ERROR "Unable to find libfmm3d under ${DMK_FMM3D_DIR}")
endif()

add_executable(benchmark_laplace_grad_fmm3d examples/benchmark_laplace_grad_fmm3d.cpp)
target_link_libraries(benchmark_laplace_grad_fmm3d PRIVATE dmk "${DMK_FMM3D_LIBRARY}")
target_include_directories(benchmark_laplace_grad_fmm3d PRIVATE "${FIDMK_ROOT}/extern/nda/c++")

add_executable(bench_pbc_accuracy bench_pbc_accuracy.cpp)
target_link_libraries(bench_pbc_accuracy PRIVATE dmk)
target_include_directories(bench_pbc_accuracy PRIVATE "${FIDMK_ROOT}/extern/nda/c++")

add_executable(bench_pbc_stagecheck bench_pbc_stagecheck.cpp)
target_link_libraries(bench_pbc_stagecheck PRIVATE dmk)
target_include_directories(bench_pbc_stagecheck PRIVATE "${FIDMK_ROOT}/extern/nda/c++")

add_executable(bench_pbc_runtime bench_pbc_runtime.cpp)
target_link_libraries(bench_pbc_runtime PRIVATE dmk OpenMP::OpenMP_CXX)
target_include_directories(bench_pbc_runtime PRIVATE "${FIDMK_ROOT}/extern/nda/c++")

get_filename_component(DMK_FMM3D_LIBRARY_DIR "${DMK_FMM3D_LIBRARY}" DIRECTORY)
set_target_properties(benchmark_laplace_grad_fmm3d PROPERTIES
  BUILD_RPATH "${DMK_FMM3D_LIBRARY_DIR}"
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/examples"
)
set_target_properties(bench_pbc_accuracy PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
)
set_target_properties(bench_pbc_stagecheck PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
)
set_target_properties(bench_pbc_runtime PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
)
