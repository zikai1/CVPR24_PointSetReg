/*
Copyright (C) 2024  Mingyang Zhao & Jingen Jiang
https://zikai1.github.io/ - migyangz@gmail.com
https://xiaowuga.github.io/ - xiaowuga@gmail.com

The code is the official implementation of our CVPR 2024 paper
"Correspondence-Free Nonrigid Point Set Registration Using Unsupervised Clustering Analysis"

Our code is under AGPL-3.0, so any downstream solution and products (including cloud services) that
include our code inside it should be open-sourced to comply with the AGPL conditions. For learning
purposes only and not for commercial use.

If you want to use it for commercial purposes, please contact us first.
*/
#include <iostream>
#include <CLI/CLI.hpp>
#include <Eigen/Core>
#include "base.h"
#include "INys.h"
#include "fuzzy_cluster_reg.h"
#include <chrono>

#include <mkl.h>
int main(int argc, char **argv) {
    struct {
        std::string src_path;
        std::string tar_path;
        std::string out_path;
    } args;

    int threads = omp_get_max_threads();;
    std::cout <<"maximal thread number = " << threads << std::endl;
//    max_threads = 10;
    Eigen::setNbThreads(threads);
    Eigen::initParallel();
    omp_set_num_threads(threads);
    mkl_set_num_threads(threads);


    CLI::App app{"CluReg Command Line"};

    app.add_option("-s,--src_path", args.src_path, "Source Point Set")->required();
    app.add_option("-t,--tar_path", args.tar_path, "Target Point Set")->required();
    app.add_option("-o,--out_path", args.out_path, "Output Point Set")->required();

    CLI11_PARSE(app, argc, argv);
    // std::string src_path = "../data/tr_reg_059.ply";
    // std::string tar_path = "../data/tr_reg_057.ply";
    Eigen::MatrixXd src_points;
    Eigen::MatrixXd tar_points;
    IO::read_3D_points(args.src_path, src_points);
    IO::read_3D_points(args.tar_path, tar_points);

    Base::PointSet src(src_points);
    Base::PointSet tar(tar_points);
    Eigen::RowVectorXd alpha;
    Base::PointSet res;

    auto start = std::chrono::high_resolution_clock::now();
    fuzzy_cluster_reg(src, tar, alpha, res);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    std::cout << "fuzzy_cluster_reg time: " << duration.count() << " s" << std::endl;
    IO::write_3D_points(args.out_path, res.points_);
    return 0;
}
