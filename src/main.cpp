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
//#include <armadillo>

//void fuzzy_cluster_reg(Eigen::MatrixXd& src, Eigen::MatrixXd& tar,
//                       double lamdba = 0.1, double beta = 0.5,
//                       double tol = 1e-5, int max_num_iter = 200) {
//    size_t nb_src = src.rows(), nb_tar = tar.rows();
//    size_t dim = src.cols();
//
//    Eigen::Vector3d src_centroid = src.colwise().mean();
//    Eigen::Vector3d tar_centroid = tar.colwise().mean();
//
//    Eigen::VectorXd centroid_diff = tar_centroid - src_centroid;
//
//    src.array().rowwise() +=  centroid_diff.transpose().array();
//
//    double sigma2 = (nb_tar * (src * src.transpose()).trace() +
//                    nb_src * (tar * tar.transpose()).trace() -
//                    2 * src.colwise().sum() * tar.colwise().sum().transpose()) / (nb_src * nb_tar * dim);
//
//    double theta = 2;
//
//    Kernel kernel = {"rbf", theta};
//    int m = ceil(0.3 * nb_src);
//
//    Eigen::MatrixXd Q;
//    improved_nystrom_low_rank_approximation(kernel, src, "k", m, Q);
//
//    size_t c = Q.cols();
//
//    Eigen::MatrixXd W = Eigen::MatrixXd::Zero(nb_src, dim);
//    int iter = 0;
//    double ntol = tol + 10;
//    double loss = 1;
//
//    Eigen::MatrixXd T = src, F = tar;
//    Eigen::MatrixXd FT = tar.transpose();
//    Eigen::MatrixXd FF = (FT.array() * FT.array()).colwise().sum();
//
//    Eigen::MatrixXd alpha = Eigen::MatrixXd::Ones(1, nb_src);
//
//    Eigen::MatrixXd onesUy = Eigen::MatrixXd::Ones(nb_src, 1);
//
//    Eigen::MatrixXd onesUx = Eigen::MatrixXd::Ones(nb_tar, 1);
//
//    Eigen::MatrixXd ident_matrix = Eigen::MatrixXd::Identity(c, c);
//
//    double eps = 1e-10;
//
//    while((ntol > tol) && (iter < max_num_iter) && (sigma2 > 1e-8)) {
//        Eigen::MatrixXd tmp;
//        sqdist2(FF, FT, T, tmp);
//        Eigen::MatrixXd fuzzy_dis = (-tmp / (sigma2 * beta)).array().exp() * alpha.array();
//    }
//
//}
int main() {
//    int max_threads = omp_get_max_threads();
//    std::cout << max_threads << std::endl;
//    Eigen::setNbThreads(max_threads);
    omp_set_num_threads(10);
    mkl_set_num_threads(10);
//    Eigen::MatrixXd a, b, res1, res2;
//
//    a.resize(2, 3);
//    a << 1,2,3,6,5,4;
//    b.resize(2,3);
//    b << 3,2,1,4,5,6;
//    sqdist_omp(a, b ,res1);
////    pdist_cityblock_omp(a, a, res1);
//    std::cout << res1 << std::endl;
//    return 0;
//    return 0;
////    a << 1, 2, 3;
//    b.resize(3, 3);
//    b << 2, 3, 4,
//         5, 6, 7,
//         8,9,10;
//    std::cout << b.diagonal() << std::endl;
//    return 0;
//    Eigen::MatrixXd ss;
//    sqdist(a, b, ss);
//    std::cout << ss << std::endl;
//    return 0;
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
//    Eigen::MatrixXd tar(1, 3);
//    tar <<  1,2,3;
////    fuzzy_cluster_reg(src, tar);
////    eff_kmeans(src, 2, 5);
//    Eigen::MatrixXd res;
////    igl::repmat(tar, 1, 3, res);
//
//    std::cout << tar.array().exp() << std::endl;
//    sqdist(tar, src, res);

//    std::cout << res << std::endl;
//    Eigen::VectorXi d(3);
//    d << 1,2,1;
//    Eigen::VectorXi I;
//    std:: cout << (d.array() == 1)<< std::endl;
////    igl::find_zero(d.array() == 1, )
//    igl::find(d.array() == 1, I);


//    Eigen::VectorXd idx(5);
//    idx << 1, 2, 1, 3, 1;
//    double j = 1; // Value to find
//
//    Eigen::VectorXi indices;
////    Eigen::VectorXi tmp = (idx.array() == j).array();
//    igl::find((idx.array() == j).eval(), indices);
//
//    // Print the indices
//    std::cout << "Indices where idx == j:" << std::endl;
//    for (int i = 0; i < indices.size(); ++i) {
//        std::cout << indices(i) << " ";
//    }
    return 0;
}
