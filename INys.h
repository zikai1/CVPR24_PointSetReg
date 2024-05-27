//
// Created by 小乌嘎 on 2024/2/21.
//

#ifndef FUZZYNONRIGID_INYS_H
#define FUZZYNONRIGID_INYS_H
#include <cmath>
#include "iostream"
#include "eff_kmeans.h"
#include <omp.h>


void pdist2_cityblock(const Eigen::MatrixXd& a, const Eigen::MatrixXd& b, Eigen::MatrixXd& res){
    int n = a.rows(), m = b.rows();
    res.resize(n, m);
    for(int i = 0; i < n; i++) {
        for(int j = 0; j < m ; j++) {
            res(i, j) = (a.row(i) - b.row(j)).array().abs().sum();
        }
    }
}

void improved_nystrom_low_rank_approximation(Base::Kernel& kernel,
                                             Eigen::MatrixXd& data,
                                             const std::string& state,
                                             int m,
                                             Eigen::MatrixXd& G) {
    size_t num = data.rows(), dim = data.cols();
    Eigen::VectorXi idx;
    Eigen::MatrixXd center;
    if(state == "k") {
        eff_kmeans(data, 5, idx, center, m);
    }
    else if(state == "r") {
        Eigen::VectorXi dex;
        igl::randperm(num, dex);
        center = data(dex.segment(0, m), Eigen::all);
    }

    Eigen::MatrixXd W, E;
    if(kernel.type_ == "pol") {
        W = center * center.transpose();
        E = data * data.transpose();
        W = W.array().pow(kernel.para_);
        E = E.array().pow(kernel.para_);
    }

    if(kernel.type_ == "rbf") {
        Eigen::MatrixXd res;
        sqdist(center, center, res);
        W = (-(res.array() / kernel.para_).array()).exp();
        sqdist(data, center, res);
        E = (-(res.array() / kernel.para_).array()).exp();
    }
//    else if(kernel.type_ == "Laplacian_L2") {
//        Eigen::MatrixXd res;
//        sqdist(center, center, res);
//        W = (-(res.array().sqrt() / kernel.para_).array()).exp();
//        sqdist(data, center, res);
//        E = (-(res.array().sqrt() / kernel.para_).array()).exp();
//    }
//    else if(kernel.type_ == "Laplacian_L1") {
////        Laplacian kernel (L1)
//        Eigen::MatrixXd res;
//        pdist2_cityblock(center, center, res);
//        W = (-res / kernel.para_).array().exp();
//        pdist2_cityblock(data, center, res);
//        E = (-res / kernel.para_).array().exp();
//    }
    Eigen::EigenSolver<Eigen::MatrixXd> eigensolver(W);
    if (eigensolver.info() != Eigen::Success) {
        std::cerr << "EigenSolver failed" << std::endl;
        return;
    }
    Eigen::VectorXd va = eigensolver.eigenvalues().real();
    Eigen::MatrixXd ve = eigensolver.eigenvectors().real();
    Eigen::VectorXi pidx;
    igl::find((va.array() > 1e-6).eval(), pidx);
    int nb_pidx = pidx.size();
    std::vector<Eigen::Triplet<double>> triplets(nb_pidx);
    Eigen::SparseMatrix<double> inVa(nb_pidx, nb_pidx);
    #pragma omp parallel for
    for(int i = 0; i < nb_pidx; i++) {
        triplets[i] = Eigen::Triplet<double>(i, i, std::pow(va[pidx[i]], -0.5));
    }

    G = E * ve(Eigen::all, pidx) * inVa;
}

#endif //FUZZYNONRIGID_INYS_H
