#include <iostream>

#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Sparse>

#include <igl/repmat.h>

#include "io.h"
#include "INys.h"



void fuzzy_cluster_reg(Eigen::MatrixXd& src, Eigen::MatrixXd& tar,
                       double lamdba = 0.1, double beta = 0.5,
                       double tol = 1e-5, int max_num_iter = 200) {
    size_t nb_src = src.rows(), nb_tar = tar.rows();
    size_t dim = src.cols();

    Eigen::Vector3d src_centroid = src.colwise().mean();
    Eigen::Vector3d tar_centroid = tar.colwise().mean();

    Eigen::VectorXd centroid_diff = tar_centroid - src_centroid;

    src.array().rowwise() +=  centroid_diff.transpose().array();

    double sigma2 = (nb_tar * (src * src.transpose()).trace() +
                    nb_src * (tar * tar.transpose()).trace() -
                    2 * src.colwise().sum() * tar.colwise().sum().transpose()) / (nb_src * nb_tar * dim);

    double theta = 2;

    Kernel kernel = {"rbf", theta};
    int m = ceil(0.3 * nb_src);

    Eigen::MatrixXd Q;
    improved_nystrom_low_rank_approximation(kernel, src, "k", m, Q);

    size_t c = Q.cols();

    Eigen::MatrixXd W = Eigen::MatrixXd::Zero(nb_src, dim);
    int iter = 0;
    double ntol = tol + 10;
    double loss = 1;

    Eigen::MatrixXd T = src, F = tar;
    Eigen::MatrixXd FT = tar.transpose();
    Eigen::MatrixXd FF = (FT.array() * FT.array()).colwise().sum();

    Eigen::MatrixXd alpha = Eigen::MatrixXd::Ones(1, nb_src);

    Eigen::MatrixXd onesUy = Eigen::MatrixXd::Ones(nb_src, 1);

    Eigen::MatrixXd onesUx = Eigen::MatrixXd::Ones(nb_tar, 1);

    Eigen::MatrixXd ident_matrix = Eigen::MatrixXd::Identity(c, c);

    double eps = 1e-10;

    while((ntol > tol) && (iter < max_num_iter) && (sigma2 > 1e-8)) {
        Eigen::MatrixXd tmp;
        sqdist2(FF, FT, T, tmp);
        Eigen::MatrixXd fuzzy_dis = (-tmp / (sigma2 * beta)).array().exp() * alpha.array();
    }

}
int main() {
    std::string src_path = " ";
    std::string tar_path = " ";
    Eigen::MatrixXd src(3, 3);
    src <<  1, 2, 3,
            5, 4, 6,
            8, 7, 2;
    Eigen::MatrixXd tar(1, 3);
    tar <<  1,2,3;
//    fuzzy_cluster_reg(src, tar);
//    eff_kmeans(src, 2, 5);
    Eigen::MatrixXd res;
//    igl::repmat(tar, 1, 3, res);

    std::cout << tar.array().exp() << std::endl;
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
