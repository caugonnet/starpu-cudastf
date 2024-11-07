cmake_minimum_required(VERSION 3.18)

# Default to building for the GPU on the current system
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES native)
endif()

set(STARPU_ROOT "/home/caugonnet/git/starpu/build/")
set(ENV{PKG_CONFIG_PATH} "${STARPU_ROOT}/lib/pkgconfig")

find_package(PkgConfig REQUIRED)
pkg_check_modules(STARPU REQUIRED starpu-1.4)


# Use CUDASTF from the CCCL project
include(cmake/CPM.cmake)

# This will automatically clone CCCL from GitHub and make the exported cmake targets available
CPMAddPackage(
    NAME CCCL
    GITHUB_REPOSITORY nvidia/cccl
    GIT_TAG main
    # The following is required to make the `CCCL::cudax` target available:
    OPTIONS "CCCL_ENABLE_UNSTABLE ON"
)

project(starpu-stf CUDA CXX)

add_executable(hello_world hello_world.cu)

# Use CUDASTF
target_link_libraries(hello_world PRIVATE CCCL::CCCL CCCL::cudax cuda)

if (CMAKE_CUDA_COMPILER)
    target_compile_options(hello_world PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
    target_compile_options(hello_world PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)
endif()

target_compile_options(hello_world PUBLIC ${STARPU_CFLAGS})
target_link_libraries(hello_world PRIVATE ${STARPU_LIBRARIES})
target_link_options(hello_world PRIVATE ${STARPU_LDFLAGS})
target_link_libraries(hello_world PRIVATE ${STARPU_ROOT}/lib/libstarpu-1.4.a)
