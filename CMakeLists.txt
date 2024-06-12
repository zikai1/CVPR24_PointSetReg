cmake_minimum_required(VERSION 3.24)
project(CluReg)

set(CMAKE_CXX_STANDARD 17)

find_package(Eigen3 CONFIG REQUIRED)
find_package(libigl CONFIG REQUIRED)
find_package(OpenMP REQUIRED)
if(OpenMP_CXX_FOUND)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

find_package(MKL CONFIG REQUIRED)

add_definitions(-DEIGEN_USE_MKL_ALL)
add_executable(CluReg src/main.cpp)
target_link_libraries(CluReg PUBLIC Eigen3::Eigen igl::igl_core OpenMP::OpenMP_CXX)
target_link_libraries(CluReg PUBLIC MKL::MKL)

add_executable(test_mkl src/test_mkl.cpp)
target_link_libraries(test_mkl PUBLIC MKL::MKL)

add_executable(test_eigen_mkl src/test_eigen_mkl.cpp)
target_link_libraries(test_eigen_mkl PUBLIC Eigen3::Eigen MKL::MKL)
