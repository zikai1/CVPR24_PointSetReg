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

#include "base.h"
#include "INys.h"
#ifndef FUZZYNONRIGID_FUZZY_CLUSTER_REG_H
#define FUZZYNONRIGID_FUZZY_CLUSTER_REG_H

void compute_fuzzy_dis_omp(Eigen::MatrixXd& fuzzy_dis, Eigen::RowVectorXd& alpha, double sigma2, double beta) {
    double div = sigma2 * beta;
    int n = fuzzy_dis.rows(), m = fuzzy_dis.cols();
#pragma omp parallel for collapse(2)
    for(int i = 0; i < n; i++) {
        for(int j = 0; j < m; j++) {
            fuzzy_dis(i,j) = exp(-fuzzy_dis(i,j) / div) * alpha[j];
        }
    }
}

void compute_sum_fuzzy_dist_omp(Eigen::MatrixXd& fuzzy_dis, Eigen::VectorXd& sum_fuzzy_dist) {
    int n = fuzzy_dis.rows(), m = fuzzy_dis.cols();
    sum_fuzzy_dist.resize(n);
#pragma omp parallel for collapse(2)
    for(int i = 0; i < n; i++) {
        sum_fuzzy_dist[i] = 0.0;
        for(int j = 0; j < m; j++) {
            #pragma omp atomic
            sum_fuzzy_dist[i] += fuzzy_dis(i, j);
        }
        sum_fuzzy_dist[i] = 1.0 / sum_fuzzy_dist[i];
    }
}

void compute_U_logU_omp(Eigen::MatrixXd& fuzzy_dis,Eigen::VectorXd& sum_fuzzy_dist, Eigen::MatrixXd& U, Eigen::MatrixXd& logU, double eps) {
    int n = fuzzy_dis.rows(), m = fuzzy_dis.cols();
//    U.resize(n, m);
//    logU.resize(n, m);
#pragma omp parallel for collapse(2)
    for(int i = 0; i < m; i++) {
        for(int j = 0; j < n; j++) {
            U(j, i) = fuzzy_dis(j, i) * sum_fuzzy_dist[j] + eps;
            logU(j, i) = log(U(j, i));
        }
    }
}

void compute_alpha_log_alpha_omp(Eigen::MatrixXd& U, Eigen::RowVectorXd& alpha, Eigen::RowVectorXd& log_alpha, double eps) {
    int n = U.rows(), m = U.cols();
//    alpha.resize(m);
//    log_alpha.resize(m);
#pragma omp parallel for collapse(2)
    for(int i = 0; i < m; i++) {
        alpha[i] = 0.0;
        for(int j = 0; j < n; j++) {
            #pragma omp atomic
            alpha[i] += U(j, i);
        }
        alpha[i] = alpha[i] / m + eps;
        log_alpha[i] = log(alpha[i]);
    }
}

void fuzzy_cluster_reg(Base::PointSet& src, Base::PointSet& tar,
                       Eigen::RowVectorXd& alpha, Base::PointSet& res) {
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

    improved_nystrom_low_rank_approximation(kernel, src_pt, m, Q);
    Eigen::MatrixXd Qt = Q.transpose();

    Eigen::MatrixXd W = Eigen::MatrixXd::Zero(nb_src, dim);
    int iter = 0,maxNumIter = 50;
    double tol = 1e-5, ntol = tol + 10, loss = 1.0;
    Eigen::MatrixXd T = src_pt;
    Eigen::MatrixXd F = tar_pt;
    Eigen::MatrixXd FT = F.transpose();
    Eigen::MatrixXd FF = FT.array().square().colwise().sum();

    double beta = 0.5, lamdba = 0.1;

    alpha = Eigen::RowVectorXd::Ones(nb_src);
    Eigen::VectorXd onesUy = Eigen::VectorXd::Ones(nb_src);
    Eigen::VectorXd onesUx = Eigen::VectorXd::Ones(nb_tar);

    size_t c = Q.cols();
    Eigen::MatrixXd IdentMatrix = Eigen::MatrixXd::Identity(c, c);

    double eps = 1e-10;
    Eigen::MatrixXd QtW(Qt.rows(), W.cols()), P(F.rows(), 3), U(F.rows(), T.rows()),logU(F.rows(), T.rows()), fuzzy_dis(F.rows(), T.rows());
    Eigen::MatrixXd A(c, c), invA(c, c);
    Eigen::MatrixXd dUtQ(Q.rows(), Q.cols()), Uttgt(F.rows(), 3);
    Eigen::RowVectorXd log_alpha(nb_src);
    Eigen::VectorXd sum_fuzzy_dist(nb_src), U1(nb_src),Ut1(nb_tar);
    while(ntol > tol && iter < maxNumIter && sigma2 > 1e-8) {

        double loss_old = loss;
        QtW.noalias() = Qt * W;

        sqdist_omp(F, T, fuzzy_dis);


        compute_fuzzy_dis_omp(fuzzy_dis, alpha, sigma2, beta);
        compute_sum_fuzzy_dist_omp(fuzzy_dis, sum_fuzzy_dist);
        compute_U_logU_omp(fuzzy_dis, sum_fuzzy_dist, U, logU, eps);

        compute_alpha_log_alpha_omp(U, alpha, log_alpha, eps);

        U1.noalias() = U * onesUy;

        Ut1.noalias() = U.transpose() * onesUx;

        Eigen::DiagonalMatrix<double, Eigen::Dynamic> dU = U1.asDiagonal();
        Eigen::DiagonalMatrix<double, Eigen::Dynamic> dUt = Ut1.asDiagonal();

        dUtQ.noalias() = dUt * Q;
        Uttgt.noalias() = U.transpose() * tar_pt;


        P.noalias() = Uttgt - dUt * src_pt;
        A.noalias() = lamdba * sigma2 * IdentMatrix + Qt * dUtQ;
        invA.noalias() =  A.inverse();


        W.noalias() = 1.0 / (lamdba * sigma2) * (P - dUtQ * (invA * (Qt * P)));

        double wdist_pt2center = fabs((FT * dU * F + T.transpose() * dUt * T - 2 * Uttgt.transpose() * T).trace());

        double H_U = (U.array() * logU.array()).sum();
        double H_alpha = nb_tar * alpha * log_alpha.transpose();
        double KL_U_alpha = H_U - H_alpha;
        loss = (1.0 / sigma2) * wdist_pt2center + nb_tar * dim * log(sigma2) +
               0.5 * lamdba  * (QtW.transpose() * QtW).trace()
               + beta * KL_U_alpha;
        ntol = abs((loss - loss_old) / loss);

        T.noalias() = src_pt + Q * (Qt * W);

        sigma2 = wdist_pt2center / (nb_tar * dim);

        std::cout << "iter: " << iter  << " sigma2: " << sigma2 << " tol: " << ntol << std::endl;
        iter++;
    }

    T = T * tar.scale_;
    T.array().rowwise() += tar.centroid_.transpose().array();
    res.set_points(T, false);

}

#endif //FUZZYNONRIGID_FUZZY_CLUSTER_REG_H
