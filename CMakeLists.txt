cmake_minimum_required(VERSION 3.24)
project(FuzzyNonRigid)

set(CMAKE_CXX_STANDARD 11)

find_package(Eigen3 CONFIG REQUIRED)

find_package(libigl CONFIG REQUIRED)

find_package(OpenMP REQUIRED)
if(OpenMP_CXX_FOUND)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

add_executable(FuzzyNonRigid main.cpp)
target_link_libraries(FuzzyNonRigid PUBLIC Eigen3::Eigen igl::igl_core OpenMP::OpenMP_CXX)