/*
* Paper:Correspondence-Free Non-Rigid Point Set Registration Using Unsupervised Clustering Analysis, CVPR 2024
* Mingyang Zhao & Jingen Jiang
* Initialized on May 27th, 2024, Hong Kong
*/

#include "base.h"

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
    U.resize(n, m);
    logU.resize(n, m);
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
    alpha.resize(m);
    log_alpha.resize(m);
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
    auto start = std::chrono::high_resolution_clock::now();

    improved_nystrom_low_rank_approximation(kernel, src_pt, m, Q);
    Eigen::MatrixXd Qt = Q.transpose();
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    std::cout << "rank_approximation time: " << duration.count() << " s" << std::endl;

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
    Eigen::MatrixXd QtW, P, U,logU, fuzzy_dis;
    Eigen::MatrixXd A, invA;
    Eigen::MatrixXd dUtQ, Uttgt;
    Eigen::RowVectorXd log_alpha;
    Eigen::VectorXd sum_fuzzy_dist(nb_src), U1,Ut1;
    double invt = 0;
    start = std::chrono::high_resolution_clock::now();

    while(ntol > tol && iter < maxNumIter && sigma2 > 1e-8) {

        double loss_old = loss;
        QtW = Qt * W;

        sqdist_omp(F, T, fuzzy_dis);


        compute_fuzzy_dis_omp(fuzzy_dis, alpha, sigma2, beta);
        compute_sum_fuzzy_dist_omp(fuzzy_dis, sum_fuzzy_dist);
        compute_U_logU_omp(fuzzy_dis, sum_fuzzy_dist, U, logU, eps);

        compute_alpha_log_alpha_omp(U, alpha, log_alpha, eps);

        U1 = U * onesUy;

        Ut1 = U.transpose() * onesUx;

        Eigen::DiagonalMatrix<double, Eigen::Dynamic> dU = U1.asDiagonal();
        Eigen::DiagonalMatrix<double, Eigen::Dynamic> dUt = Ut1.asDiagonal();

        dUtQ = dUt * Q;
        Uttgt = U.transpose() * tar_pt;


        P = Uttgt - dUt * src_pt;
        A = lamdba * sigma2 * IdentMatrix + Qt * dUtQ;
        auto ss = std::chrono::high_resolution_clock::now();
        Eigen::MatrixXd cc = Qt * dUtQ;
        auto ee = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> dd = ee - ss;
        invt += dd.count();
        invA =  A.inverse();


        W = 1.0 / (lamdba * sigma2) * (P - dUtQ * (invA * (Qt * P)));

        double wdist_pt2center = fabs((FT * dU * F + T.transpose() * dUt * T - 2 * Uttgt.transpose() * T).trace());

        double H_U = (U.array() * logU.array()).sum();
        double H_alpha = nb_tar * alpha * log_alpha.transpose();
        double KL_U_alpha = H_U - H_alpha;
        loss = (1.0 / sigma2) * wdist_pt2center + nb_tar * dim * log(sigma2) +
               0.5 * lamdba  * (QtW.transpose() * QtW).trace()
               + beta * KL_U_alpha;
        ntol = abs((loss - loss_old) / loss);

        T = src_pt + Q * (Qt * W);

        sigma2 = wdist_pt2center / (nb_tar * dim);

        std::cout << "iter: " << iter  << " sigma2: " << sigma2 << " tol: " << ntol << std::endl;
 //        break;
        iter++;
    }
    end = std::chrono::high_resolution_clock::now();
    duration = end - start;
    std::cout << "rank_approximation time: " << duration.count() << " s" << std::endl;
    std::cout << "invt time: " << invt << " s" << std::endl;
    res.set_points(T, false);
}

#endif //FUZZYNONRIGID_FUZZY_CLUSTER_REG_H
