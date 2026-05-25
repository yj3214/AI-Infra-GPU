cmake_minimum_required(VERSION 3.20)

project(myai_gpu
    VERSION 0.1.0
    DESCRIPTION "Minimal CMake SDK scaffold for a C++20 GPU inference library"
    LANGUAGES CXX
)

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
)

target_compile_features(myai_gpu PUBLIC cxx_std_20)

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

enable_testing()
add_test(NAME simple_demo_runs COMMAND simple_demo)
