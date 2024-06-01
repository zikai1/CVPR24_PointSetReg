/*
* Paper:Correspondence-Free Non-Rigid Point Set Registration Using Unsupervised Clustering Analysis, CVPR 2024
* Mingyang Zhao & Jingen Jiang
* Initialized on May 27th, 2024, Hong Kong
*/

#include "base.h"

#ifndef FUZZYNONRIGID_FUZZY_CLUSTER_REG_H
#define FUZZYNONRIGID_FUZZY_CLUSTER_REG_H

void fuzzy_cluster_reg(Base::PointSet& src, Base::PointSet& tar,
                       Eigen::VectorXd& alpha, Base::PointSet& res) {
    int nb_src = src.nb_points_;
    int nb_tar = tar.nb_points_;
    int dim = src.dim_;
    Eigen::MatrixXd& src_pt = src.points_;
    Eigen::MatrixXd& tar_pt = tar.points_;
    double sigma2 = (nb_tar * (src_pt * src_pt.transpose()).trace() +
                nb_src * (tar_pt * tar_pt.transpose()).trace() -
                2 * src_pt.colwise().sum() * tar_pt.colwise().sum().transpose()) / (nb_src * nb_tar * dim);

    double theta = 0.5;
    Base::Kernel kernel = {"rbf_l1", theta};

    double m = ceil(0.3 * nb_src);
    Eigen::MatrixXd Q;
    auto start = std::chrono::high_resolution_clock::now();

    improved_nystrom_low_rank_approximation(kernel, src_pt, m, Q);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    std::cout << "rank_approximation time: " << duration.count() << " s" << std::endl;

    Eigen::MatrixXd W = Eigen::MatrixXd::Zero(nb_src, dim);
    int iter = 0,maxNumIter = 50;
    double tol = 1e-5, ntol = tol + 10, loss = 1;
    Eigen::MatrixXd T = src_pt;
    Eigen::MatrixXd F = tar_pt;
    Eigen::MatrixXd FT = F.transpose();

}

#endif //FUZZYNONRIGID_FUZZY_CLUSTER_REG_H
