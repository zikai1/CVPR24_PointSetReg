//
// Created by 小乌嘎 on 2024/2/21.
//

#ifndef FUZZYNONRIGID_INYS_H
#define FUZZYNONRIGID_INYS_H
#include <cmath>
#include "iostream"
#include "kmeans.h"
#include <omp.h>

#include <Eigen/Core>
#include <Eigen/Dense>

//#include <MatlabEngine.hpp>
//#include <MatlabDataArray.hpp>

#include <Eigen/Eigenvalues>



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




void improved_nystrom_low_rank_approximation(Base::Kernel& kernel,
                                             Eigen::MatrixXd& data,
                                             const std::string& state,
                                             int m,
                                             Eigen::MatrixXd& G) {
    size_t num = data.rows(), dim = data.cols();
    Eigen::VectorXi idx;
    Eigen::MatrixXd center;

    auto start = std::chrono::high_resolution_clock::now();
//    kmeans(data, 5, idx, center, m);

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
//    std::cout << "kmeans time: " << duration.count() << " s" << std::endl;

    start = std::chrono::high_resolution_clock::now();

    elkan_kmeans(data, 5, idx, center, m);

    end = std::chrono::high_resolution_clock::now();
    duration = end - start;
    std::cout << "elkan_kmeans time: " << duration.count() << " s" << std::endl;


    Eigen::MatrixXd W, E;

    if(kernel.type_ == "rbf_l1") {
        Eigen::MatrixXd res;
        pdist_cityblock_omp(center, center, res);
        W = (-(res.array() / kernel.para_).array()).exp();
        pdist_cityblock_omp(data, center, res);
        E = (-(res.array() / kernel.para_).array()).exp();
    }


    start = std::chrono::high_resolution_clock::now();
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigensolver(W);
    if (eigensolver.info() != Eigen::Success) {
        std::cerr << "EigenSolver failed" << std::endl;
        return;
    }
    end = std::chrono::high_resolution_clock::now();
    duration = end - start;
    std::cout << "EigenSolver time: " << duration.count() << " s" << std::endl;



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
    inVa.setFromTriplets(triplets.begin(),triplets.end());

    start = std::chrono::high_resolution_clock::now();
    G = E * ve(Eigen::all, pidx) * inVa;
    end = std::chrono::high_resolution_clock::now();
    duration = end - start;
    std::cout << "Easd time: " << duration.count() << " s" << std::endl;
}

#endif //FUZZYNONRIGID_INYS_H
