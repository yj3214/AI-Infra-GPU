cmake_minimum_required(VERSION 3.20)

project(myai_gpu
    VERSION 0.1.0
    DESCRIPTION "Minimal CMake SDK scaffold for a C++20 GPU inference library"
    LANGUAGES CXX
)

option(MYAI_GPU_BUILD_EXAMPLES "Build example executables" ON)
option(MYAI_GPU_BUILD_TESTS "Build tests" ON)

find_package(CUDAToolkit REQUIRED)
set(OpenCV_DIR "/usr/local/lib/cmake/opencv4")
find_package(OpenCV REQUIRED)

if(MYAI_GPU_BUILD_TESTS)
    find_package(GTest QUIET)

    if(NOT GTest_FOUND)
        include(FetchContent)
        FetchContent_Declare(
            googletest
            URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.zip
            DOWNLOAD_EXTRACT_TIMESTAMP TRUE
        )
        FetchContent_MakeAvailable(googletest)
    endif()

    if(TARGET GTest::gtest)
        set(MYAI_GTEST_TARGET GTest::gtest)
    else()
        set(MYAI_GTEST_TARGET gtest)
    endif()
endif()

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
    src/classifier.cpp
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
        ${OpenCV_LIBS}
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

if(MYAI_GPU_BUILD_EXAMPLES)
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/example/simple/main.cpp")
        add_executable(simple_demo
            example/simple/main.cpp
        )

        target_link_libraries(simple_demo
            PRIVATE
                myai_gpu::myai_gpu
        )
    endif()

    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/example/cls/main.cpp")
        add_executable(classification_demo
            example/cls/main.cpp
        )

        target_link_libraries(classification_demo
            PRIVATE
                myai_gpu::myai_gpu
                ${OpenCV_LIBS}
        )
    endif()
endif()

if(MYAI_GPU_BUILD_TESTS)
    add_executable(model_gtest
        test/model_gtest.cpp
    )

    target_link_libraries(model_gtest
        PRIVATE
            myai_gpu::myai_gpu
            ${MYAI_GTEST_TARGET}
    )

    add_executable(classifier_gtest
        test/classifier_gtest.cpp
    )

    target_link_libraries(classifier_gtest
        PRIVATE
            myai_gpu::myai_gpu
            ${OpenCV_LIBS}
            ${MYAI_GTEST_TARGET}
    )
endif()

enable_testing()

if(MYAI_GPU_BUILD_EXAMPLES)
    if(TARGET simple_demo)
        add_test(NAME simple_demo_runs COMMAND simple_demo)
        set_tests_properties(simple_demo_runs PROPERTIES
            WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        )
    endif()
    if(TARGET classification_demo)
        add_test(NAME classification_demo_runs COMMAND classification_demo)
        set_tests_properties(classification_demo_runs PROPERTIES
            WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        )
    endif()
endif()

if(MYAI_GPU_BUILD_TESTS)
    add_test(
        NAME model_gtest_runs
        COMMAND model_gtest
            --model_path=${CMAKE_CURRENT_SOURCE_DIR}/model.plan
            --gtest_output=xml:${CMAKE_CURRENT_BINARY_DIR}/model_gtest_report.xml
    )
    set_tests_properties(model_gtest_runs PROPERTIES
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    )
    add_test(
        NAME classifier_gtest_runs
        COMMAND classifier_gtest
            --model_path=${CMAKE_CURRENT_SOURCE_DIR}/model.plan
            --image_path=${CMAKE_CURRENT_SOURCE_DIR}/imgs/cat.jpg
            --config_path=${CMAKE_CURRENT_SOURCE_DIR}/config/cls_config.json
            --gtest_output=xml:${CMAKE_CURRENT_BINARY_DIR}/classifier_gtest_report.xml
    )
    set_tests_properties(classifier_gtest_runs PROPERTIES
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    )
endif()
