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

find_package(MKL CONFIG REQUIRED)

#find_package(Matlab REQUIRED COMPONENTS MAIN_PROGRAM)

#find_package(spectra CONFIG REQUIRED)



add_definitions(-DEIGEN_USE_MKL_ALL)
add_executable(CluReg src/main.cpp)
#target_include_directories(CluReg PUBLIC ${Matlab_INCLUDE_DIRS})
#target_link_libraries(CluReg PUBLIC ${Matlab_LIBRARIES})
target_link_libraries(CluReg PUBLIC Eigen3::Eigen igl::igl_core OpenMP::OpenMP_CXX)
target_link_libraries(CluReg PUBLIC MKL::MKL)

#add_executable(test_matlab test_matlab.cpp)
#target_include_directories(test_matlab PUBLIC ${Matlab_INCLUDE_DIRS})
#target_link_libraries(test_matlab PUBLIC ${Matlab_LIBRARIES})


#message(${MKL_ROOT})
#message(asdsad)
#get_target_property(MKL_PROPERTIES MKL::MKL PROPERTIES)
#foreach(prop ${MKL_PROPERTIES})
#    get_target_property(value MKL::MKL ${prop})
#    message("${prop} = ${value}")
#endforeach()
#get_target_property(MKL_INCLUDE_DIRS MKL::MKL INTERFACE_INCLUDE_DIRECTORIES)
#message("MKL include directories: ${MKL_INCLUDE_DIRS}")
#
#get_target_property(MKL_LIBRARIES MKL::MKL INTERFACE_LINK_LIBRARIES)
#message("MKL link libraries: ${MKL_LIBRARIES}")



add_executable(test_mkl src/test_mkl.cpp)
target_link_libraries(test_mkl PUBLIC MKL::MKL)



add_executable(test_eigen_mkl src/test_eigen_mkl.cpp)
target_link_libraries(test_eigen_mkl PUBLIC Eigen3::Eigen MKL::MKL)
