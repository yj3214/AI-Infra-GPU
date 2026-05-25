cmake_minimum_required(VERSION 3.20)

project(myai_gpu
    VERSION 0.1.0
    DESCRIPTION "Minimal CMake SDK scaffold for a C++20 GPU inference library"
    LANGUAGES CXX
)

find_package(CUDAToolkit REQUIRED)
set(OpenCV_DIR "/usr/local/lib/cmake/opencv4")
find_package(OpenCV REQUIRED)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_path(TENSORRT_INCLUDE_DIR
    NAMES NvInfer.h
    PATHS /usr/include/x86_64-linux-gnu /usr/include /usr/local/include
)

find_library(TENSORRT_NVINFER_LIBRARY
    NAMES nvinfer
    PATHS /usr/lib/x86_64-linux-gnu /usr/lib /usr/local/lib
)

find_library(TENSORRT_NVONNXPARSER_LIBRARY
    NAMES nvonnxparser
    PATHS /usr/lib/x86_64-linux-gnu /usr/lib /usr/local/lib
)

find_library(TENSORRT_NVINFER_PLUGIN_LIBRARY
    NAMES nvinfer_plugin
    PATHS /usr/lib/x86_64-linux-gnu /usr/lib /usr/local/lib
)

if(NOT TENSORRT_INCLUDE_DIR OR NOT TENSORRT_NVINFER_LIBRARY OR NOT TENSORRT_NVONNXPARSER_LIBRARY OR NOT TENSORRT_NVINFER_PLUGIN_LIBRARY)
    message(FATAL_ERROR "TensorRT headers or libraries were not found")
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_library(myai_gpu SHARED
    src/model.cpp
)

add_library(myai_gpu::myai_gpu ALIAS myai_gpu)

target_include_directories(myai_gpu
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
        ${TENSORRT_INCLUDE_DIR}
    ${CUDAToolkit_INCLUDE_DIRS}
)

target_compile_features(myai_gpu PUBLIC cxx_std_20)

target_link_libraries(myai_gpu
    PUBLIC
        CUDA::cudart
    PRIVATE
        ${TENSORRT_NVINFER_LIBRARY}
        ${TENSORRT_NVONNXPARSER_LIBRARY}
        ${TENSORRT_NVINFER_PLUGIN_LIBRARY}
)

set_target_properties(myai_gpu PROPERTIES
    OUTPUT_NAME myai_gpu
    VERSION ${PROJECT_VERSION}
    SOVERSION ${PROJECT_VERSION_MAJOR}
)

add_executable(simple_demo
    example/simple/main.cpp
)

target_link_libraries(simple_demo
    PRIVATE
        myai_gpu::myai_gpu
)

add_executable(classification_demo
    example/cls/main.cpp
)

target_link_libraries(classification_demo
    PRIVATE
        myai_gpu::myai_gpu
    ${OpenCV_LIBS}
)

enable_testing()
add_test(NAME simple_demo_runs COMMAND simple_demo)
set_tests_properties(simple_demo_runs PROPERTIES
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)
add_test(NAME classification_demo_runs COMMAND classification_demo)
set_tests_properties(classification_demo_runs PROPERTIES
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)
