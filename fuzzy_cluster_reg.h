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
    Base::Kernel kernel = {"rbf", theta};

    double m = ceil(0.3 * nb_src);
    Eigen::MatrixXd Q;
    improved_nystrom_low_rank_approximation(kernel, src_pt, "k", m, Q);
//    std::cout << Q.rows() << ' ' << Q.cols() << std::endl;
}

#endif //FUZZYNONRIGID_FUZZY_CLUSTER_REG_H
