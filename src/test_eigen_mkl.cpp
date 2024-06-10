//
// Created by 13900K on 2024/5/29.
//
#include <Eigen/Dense>
#include <iostream>
#include <chrono>
#include <mkl.h>
int main() {
#ifdef EIGEN_USE_MKL
    std::cout << "Eigen is using MKL for acceleration." << std::endl;
#else
    std::cout << "Eigen is not using MKL." << std::endl;
#endif

    mkl_set_num_threads(10);
    Eigen::MatrixXd A = Eigen::MatrixXd::Random(5000, 5000);
    Eigen::MatrixXd B = Eigen::MatrixXd::Random(5000, 5000);

    auto start = std::chrono::high_resolution_clock::now();
    Eigen::MatrixXd C = A * B;
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    std::cout << "time: " << duration.count() << " s" << std::endl;
//    std::cout << "Result matrix: " << C << std::endl;

    return 0;
}