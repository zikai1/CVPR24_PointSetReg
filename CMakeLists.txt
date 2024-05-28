cmake_minimum_required(VERSION 3.24)
project(FuzzyNonRigid)

set(CMAKE_CXX_STANDARD 17)

#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")

find_package(Eigen3 CONFIG REQUIRED)
find_package(libigl CONFIG REQUIRED)
find_package(OpenMP REQUIRED)
if(OpenMP_CXX_FOUND)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()
find_package(Matlab REQUIRED COMPONENTS MAIN_PROGRAM)

#find_package(spectra CONFIG REQUIRED)

find_package(MKL CONFIG REQUIRED)


add_executable(FuzzyNonRigid main.cpp)
target_include_directories(FuzzyNonRigid PUBLIC ${Matlab_INCLUDE_DIRS})
target_link_libraries(FuzzyNonRigid PUBLIC ${Matlab_LIBRARIES})
target_link_libraries(FuzzyNonRigid PUBLIC Eigen3::Eigen igl::igl_core OpenMP::OpenMP_CXX)
#target_link_libraries(FuzzyNonRigid PUBLIC Spectra::Spectra)

add_executable(test_matlab test_matlab.cpp)
target_include_directories(test_matlab PUBLIC ${Matlab_INCLUDE_DIRS})
target_link_libraries(test_matlab PUBLIC ${Matlab_LIBRARIES})

message(STATUS "MKL found: ${MKL_FOUND}")
message(STATUS "MKL include directory: ${MKL_INCLUDE_DIR}")
message(STATUS "MKL library directories: ${MKL_LIBRARIES}")
add_executable(test_mkl test_mkl.cpp)
target_link_libraries(test_mkl PUBLIC MKL::MKL)
