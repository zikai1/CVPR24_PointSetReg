//
// Created by 小乌嘎 on 2024/2/21.
//

#ifndef FUZZYNONRIGID_INYS_H
#define FUZZYNONRIGID_INYS_H
#include <cmath>
#include "iostream"
#include "eff_kmeans.h"
#include <omp.h>

#include <Eigen/Core>
#include <Eigen/Dense>

#include <MatlabEngine.hpp>
#include <MatlabDataArray.hpp>

#include <Eigen/Eigenvalues>

#include <Spectra/SymEigsSolver.h>
#include <Spectra/SymEigsShiftSolver.h>
#include <Spectra/MatOp/DenseSymMatProd.h>
#include <Spectra/MatOp/DenseSymShiftSolve.h>


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


matlab::data::TypedArray<double> eigenToMatlab(Eigen::MatrixXd& eigenMat) {
    matlab::data::ArrayFactory factory;
    auto matlabArray = factory.createArray<double>({(size_t)eigenMat.rows(), (size_t)eigenMat.cols()});
    for (Eigen::Index i = 0; i < eigenMat.rows(); ++i) {
        for (Eigen::Index j = 0; j < eigenMat.cols(); ++j) {
            matlabArray[i][j] = eigenMat(i, j);
        }
    }
    return matlabArray;
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
        kmeans(data, 5, idx, center, m);
    }
    else if(state == "r") {
        Eigen::VectorXi dex;
        igl::randperm(num, dex);
        center = data(dex.segment(0, m), Eigen::all);
    }

    Eigen::MatrixXd W, E;

    if(kernel.type_ == "rbf_l1") {
        Eigen::MatrixXd res;
        pdist_cityblock_omp(center, center, res);
        W = (-(res.array() / kernel.para_).array()).exp();
        pdist_cityblock_omp(data, center, res);
        E = (-(res.array() / kernel.para_).array()).exp();
    }


    auto start = std::chrono::high_resolution_clock::now();
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigensolver(W);
//    eigensolver.compute(W);
    if (eigensolver.info() != Eigen::Success) {
        std::cerr << "EigenSolver failed" << std::endl;
        return;
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double>  duration = end - start;
    std::cout << "Eigen Eigenvalue decomposition time: " << duration.count() << " s" << std::endl;

    start = std::chrono::high_resolution_clock::now();
    int ss = W.rows();
    std::cout << ss << std::endl;
    Spectra::DenseSymMatProd<double> op(W);

    Spectra::SymEigsSolver< Spectra::DenseSymMatProd<double>> eigs(op, ss, ss+1);
    eigs.init();
    int nconv = eigs.compute(Spectra::SortRule::LargestMagn, 1000, 1e-15, Spectra::SortRule::SmallestAlge);
    int niter = eigs.num_iterations();
    int nops = eigs.num_operations();
    end = std::chrono::high_resolution_clock::now();
    duration = end - start;
    std::cout << "Spectra Eigenvalue decomposition time: " << duration.count() << " s" << std::endl;
    std::cout << "niter =" <<niter<< std::endl;

    start = std::chrono::high_resolution_clock::now();
    std::unique_ptr<matlab::engine::MATLABEngine> matlabPtr = matlab::engine::startMATLAB();
    end = std::chrono::high_resolution_clock::now();
    duration = end - start;
    std::cout << "StartMATLAB time: " << duration.count() << " s" << std::endl;
    matlab::data::TypedArray<double> matlabArray = eigenToMatlab(W);


    start = std::chrono::high_resolution_clock::now();
    matlabPtr->setVariable(u"A", std::move(matlabArray));
    matlabPtr->eval(u"[V, D] = eig(A);");
    end = std::chrono::high_resolution_clock::now();
    duration = end - start;
    std::cout << "Matlab Eigenvalue decomposition time: " << duration.count() << " s" << std::endl;

    Eigen::VectorXd va = eigensolver.eigenvalues().real();
    std::cout << va.size() <<std::endl;
    Eigen::MatrixXd ve = eigensolver.eigenvectors().real();
    duration = end - start;
//    std::cout << "time: " << duration.count() << " s" << std::endl;
//    for(int i = 0; i < va.size(); i++) {
//        std::cout << va[i] <<std::endl;
//    }
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
