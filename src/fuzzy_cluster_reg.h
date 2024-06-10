/*
* Paper:Correspondence-Free Non-Rigid Point Set Registration Using Unsupervised Clustering Analysis, CVPR 2024
* Mingyang Zhao & Jingen Jiang
* Initialized on May 27th, 2024, Hong Kong
*/

#include "base.h"

#ifndef FUZZYNONRIGID_FUZZY_CLUSTER_REG_H
#define FUZZYNONRIGID_FUZZY_CLUSTER_REG_H

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
    std::cout << "sigma2=" << sigma2 << std::endl;
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
    Eigen::MatrixXd FF = FT.array().square().colwise().sum();

    double beta = 0.5, lamdba = 0.1;

    alpha = Eigen::RowVectorXd::Ones(nb_src);
    Eigen::VectorXd onesUy = Eigen::VectorXd::Ones(nb_src);
    Eigen::VectorXd onesUx = Eigen::VectorXd::Ones(nb_tar);

    size_t c = Q.cols();
    Eigen::MatrixXd IdentMatrix = Eigen::MatrixXd::Identity(c, c);

    double eps = 1e-10;

    while(ntol > tol && iter < maxNumIter && sigma2 > 1e-8) {

        double loss_old = loss;
        Eigen::MatrixXd QtW = Q.transpose() * W;
        Eigen::MatrixXd fuzzy_dis;
        sqdist_omp(F, T, fuzzy_dis);
        fuzzy_dis = (-fuzzy_dis / (sigma2 * beta)).array().exp().array().rowwise() * alpha.array();
        std::cout << fuzzy_dis.rows() << ' ' << fuzzy_dis.cols() << std::endl;

        Eigen::RowVectorXd sum_fuzzy_dist = 1.0 / fuzzy_dis.rowwise().sum().array();
//        std::cout << sum_fuzzy_dist.rows() << ' ' << sum_fuzzy_dist.cols() << std::endl;
        Eigen::MatrixXd U = fuzzy_dis.array().rowwise() * sum_fuzzy_dist.array();
        std::cout << U.rows() << ' ' << U.cols() << std::endl;
        U = U.array() + eps;
        std::cout << U.rows() << ' ' << U.cols() << std::endl;
        Eigen::MatrixXd logU = U.array().log();
        alpha = U.colwise().sum() / nb_tar;

        alpha = alpha.array() + eps;
        Eigen::RowVectorXd log_alpha = alpha.array().log();

        Eigen::MatrixXd U1 = U * onesUy;

        Eigen::MatrixXd Ut1 = U.transpose() * onesUx;

        Eigen::DiagonalMatrix<double, Eigen::Dynamic> dU = U1.asDiagonal();
        Eigen::DiagonalMatrix<double, Eigen::Dynamic> dUt = Ut1.asDiagonal();

        Eigen::SparseMatrix<double> dUtQ = (dUt * Q).sparseView();
        Eigen::MatrixXd Uttgt = U.transpose() * tar_pt;
//        std::cout <<"Uttgt shape="<< Uttgt.rows() << ' ' << Uttgt.cols() << std::endl;
//        std::cout << dUt.rows() << ' ' << dUt.cols() << std::endl;
//        Eigen::MatrixXd P = Uttgt - dUt * src_pt;
        Eigen::MatrixXd P = dUt * src_pt;
        auto start = std::chrono::high_resolution_clock::now();
        Eigen::SparseMatrix<double> A = (lamdba * sigma2 * IdentMatrix + Q.transpose() * dUtQ).sparseView();
        Eigen::SparseLU<Eigen::SparseMatrix<double>> solver;
        solver.compute(A);

        if (solver.info() != Eigen::Success) {
            std::cerr << "Decomposition failed" << std::endl;
        }



        // 构建单位矩阵 I
        Eigen::SparseMatrix<double> I(A.rows(), A.cols());
        I.setIdentity();

        // 计算稀疏矩阵 A 的逆
        Eigen::MatrixXd AinvQ = solver.solve(Q.transpose());
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> duration = end - start;
        std::cout << "elkan_kmeans time: " << duration.count() << " s" << std::endl;

        W = 1.0 / (lamdba * sigma2) * (P - dUtQ * (AinvQ * P));

        double wdist_pt2center = fabs((FT * dU * F + T.transpose() * dUt * T - 2 * Uttgt.transpose() * T).trace());
//        std::cout << wdist_pt2center << std::endl;
        double H_U = (U.array() * logU.array()).sum();
        double H_alpha = nb_tar * alpha * log_alpha.transpose();
        double KL_U_alpha = H_U - H_alpha;
        loss = (1.0 / sigma2) * wdist_pt2center + nb_tar * dim * log(sigma2) +
               0.5 * lamdba  * (QtW.transpose() * QtW).trace()
               + beta * KL_U_alpha;
        std::cout <<  beta * KL_U_alpha << std::endl;
        std::cout << 0.5 * lamdba  * (QtW.transpose() * QtW).trace() << std::endl;
        std::cout << "loss=" << loss << std::endl;
        ntol = abs((loss - loss_old) / loss);

        T = src_pt + Q * (Q.transpose() * W);

        sigma2 = wdist_pt2center / (nb_tar * dim);
//        break;
        iter++;
    }

}

#endif //FUZZYNONRIGID_FUZZY_CLUSTER_REG_H
