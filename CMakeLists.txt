cmake_minimum_required(VERSION 3.24)
project(CluReg)

set(CMAKE_CXX_STANDARD 17)

find_package(Eigen3 CONFIG REQUIRED)
#find_package(libigl CONFIG REQUIRED)
find_package(OpenMP REQUIRED)
#CLI11
find_package(CLI11 CONFIG REQUIRED)
if(OpenMP_CXX_FOUND)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

find_package(MKL CONFIG REQUIRED)

add_definitions(-DEIGEN_USE_MKL_ALL)
add_executable(CluReg src/main.cpp)
target_link_libraries(CluReg PUBLIC Eigen3::Eigen CLI11::CLI11  OpenMP::OpenMP_CXX)
target_link_libraries(CluReg PUBLIC MKL::MKL)

