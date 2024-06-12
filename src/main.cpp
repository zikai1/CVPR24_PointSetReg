#include <iostream>

#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Sparse>

#include <igl/repmat.h>

#include "base.h"
#include "INys.h"
#include "fuzzy_cluster_reg.h"
#include <chrono>

#include <mkl.h>
int main() {
    int max_threads = omp_get_max_threads();
    std::cout << max_threads << std::endl;
//    max_threads = 10;
    Eigen::setNbThreads(max_threads);
    Eigen::initParallel();


    omp_set_num_threads(max_threads);
//    mkl_set_num_threads(10);
    std::string src_path = "../data/tr_reg_059.ply";
    std::string tar_path = "../data/tr_reg_057.ply";
    Eigen::MatrixXd src_points;
    Eigen::MatrixXd tar_points;
    IO::read_3D_points(src_path, src_points);
    IO::read_3D_points(tar_path, tar_points);

    Base::PointSet src(src_points);
    Base::PointSet tar(tar_points);
    Eigen::RowVectorXd alpha;
    Base::PointSet res;

    auto start = std::chrono::high_resolution_clock::now();
    fuzzy_cluster_reg(src, tar, alpha, res);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    std::cout << "fuzzy_cluster_reg time: " << duration.count() << " s" << std::endl;
    IO::write_3D_points("../out.ply", res.points_);
    return 0;
}
