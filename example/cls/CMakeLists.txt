cmake_minimum_required(VERSION 3.20)

project(myai_gpu_cls_sdk_sample
    VERSION 0.1.0
    DESCRIPTION "Standalone classifier sample that links prebuilt myai_gpu SDK"
    LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(MYAI_GPU_SDK_ROOT "${CMAKE_CURRENT_LIST_DIR}/../.." CACHE PATH "Path to AI-Infra-GPU SDK root")
get_filename_component(MYAI_GPU_SDK_ROOT "${MYAI_GPU_SDK_ROOT}" ABSOLUTE)

set(OpenCV_DIR "/usr/local/lib/cmake/opencv4" CACHE PATH "Path to OpenCV CMake config")
find_package(OpenCV REQUIRED)
find_package(CUDAToolkit REQUIRED)

set(MYAI_GPU_INCLUDE_DIR "${MYAI_GPU_SDK_ROOT}/include" CACHE PATH "Path to myai_gpu SDK headers")
set(MYAI_GPU_LIBRARY "${MYAI_GPU_SDK_ROOT}/build/libmyai_gpu.so" CACHE FILEPATH "Path to prebuilt libmyai_gpu.so")

if(NOT EXISTS "${MYAI_GPU_INCLUDE_DIR}/classifier.hpp")
    message(FATAL_ERROR "classifier.hpp not found in ${MYAI_GPU_INCLUDE_DIR}")
endif()

if(NOT EXISTS "${MYAI_GPU_LIBRARY}")
    message(FATAL_ERROR "libmyai_gpu.so not found: ${MYAI_GPU_LIBRARY}. Build SDK first with: cmake --build ${MYAI_GPU_SDK_ROOT}/build --target myai_gpu")
endif()

add_library(myai_gpu::myai_gpu SHARED IMPORTED)
set_target_properties(myai_gpu::myai_gpu PROPERTIES
    IMPORTED_LOCATION "${MYAI_GPU_LIBRARY}"
    INTERFACE_INCLUDE_DIRECTORIES "${MYAI_GPU_INCLUDE_DIR}"
)

add_executable(cls_sdk_sample main.cpp)

target_link_libraries(cls_sdk_sample
    PRIVATE
        myai_gpu::myai_gpu
        ${OpenCV_LIBS}
        CUDA::cudart
)

set_target_properties(cls_sdk_sample PROPERTIES
    BUILD_RPATH "${MYAI_GPU_SDK_ROOT}/build"
    INSTALL_RPATH "${MYAI_GPU_SDK_ROOT}/build"
)
