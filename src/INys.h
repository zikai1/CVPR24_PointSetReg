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

#ifndef FUZZYNONRIGID_INYS_H
#define FUZZYNONRIGID_INYS_H
#include <cmath>
#include "iostream"
#include "kmeans.h"
#include <omp.h>

#include <Eigen/Core>
#include <Eigen/Dense>

#include <Eigen/Eigenvalues>
#include <Eigen/Sparse>


void pdist_cityblock_omp(const Eigen::MatrixXd& a, const Eigen::MatrixXd& b, Eigen::MatrixXd& res){
    int n = a.rows(), m = b.rows();
    res.resize(n, m);
    #pragma omp parallel for collapse(2) num_threads(4)
    for(int i = 0; i < n; i++) {
        for(int j = 0; j < m ; j++) {
//            res(i, j) = (a.row(i) - b.row(j)).array().abs().sum();
            res(i, j) = (a.row(i) - b.row(j)).cwiseAbs().sum();
        }
    }
}

void pdist2_cityblock_matrix(const Eigen::MatrixXd& a, const Eigen::MatrixXd& b, Eigen::MatrixXd& res){
    int n = a.rows(), m = b.rows();
    res.resize(n, m);
    Eigen::MatrixXd a_repeated = a.replicate(1, m);
    Eigen::MatrixXd b_repeated = b.transpose().replicate(n, 1);


    // Compute the Manhattan distances
    res = (a_repeated - b_repeated).cwiseAbs().rowwise().sum().reshaped(n, m);
}

void find_greater(const Eigen::VectorXd& data, Eigen::VectorXi& idx, double val) {
    int len = data.size(), ct = 0;
    idx.resize(len);
    for(int i = 0; i < len; i++) {
        if(data[i] > val) {
            idx[ct++] = i;
        }
    }
    idx.conservativeResize(ct);
}

void improved_nystrom_low_rank_approximation(Base::Kernel& kernel,
                                             Eigen::MatrixXd& data,
                                             int m,
                                             Eigen::MatrixXd& G) {
    size_t num = data.rows(), dim = data.cols();
    Eigen::VectorXi idx;
    Eigen::MatrixXd center;


    elkan_kmeans(data, 5, idx, center, m);
    // kmeans(data, 5, idx, center, m);

    Eigen::MatrixXd W, E;

    if(kernel.type_ == "rbf_l1") {
        Eigen::MatrixXd res;
        pdist_cityblock_omp(center, center, res);
        W = (-(res.array() / kernel.para_).array()).exp();
        pdist_cityblock_omp(data, center, res);
        E = (-(res.array() / kernel.para_).array()).exp();
    }


    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigensolver(W);
    if (eigensolver.info() != Eigen::Success) {
        std::cerr << "EigenSolver failed" << std::endl;
        return;
    }

    Eigen::VectorXd va = eigensolver.eigenvalues().real();
    Eigen::MatrixXd ve = eigensolver.eigenvectors().real();


    Eigen::VectorXi pidx;

    find_greater(va, pidx, 1e-6);
    int nb_pidx = pidx.size();
    std::vector<Eigen::Triplet<double>> triplets(nb_pidx);
    Eigen::SparseMatrix<double> inVa(nb_pidx, nb_pidx);

    #pragma omp parallel for
    for(int i = 0; i < nb_pidx; i++) {
        triplets[i] = Eigen::Triplet<double>(i, i, std::pow(va[pidx[i]], -0.5));
    }
    inVa.setFromTriplets(triplets.begin(),triplets.end());

    G = (E * ve(Eigen::all, pidx) * inVa);
}

#endif //FUZZYNONRIGID_INYS_H
