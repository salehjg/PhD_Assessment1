cmake_minimum_required(VERSION 3.10)
message("CMake Version: ${CMAKE_VERSION}")
project(CppImpl LANGUAGES CXX)

include_directories(
        ${CMAKE_SOURCE_DIR}/inc
        ${CMAKE_SOURCE_DIR}/submodules/argparse
        ${CMAKE_SOURCE_DIR}/submodules/spdlog/include
)

add_executable(Inference
        src/Main.cpp
        src/GlobalHelpers.cpp
        src/cnpy.cpp
        src/CClassifier.cpp
        src/CTensorBase.cpp)
target_link_libraries(Inference PRIVATE z stdc++fs spdlog)
add_subdirectory(${CMAKE_SOURCE_DIR}/submodules/spdlog)
