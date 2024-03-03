//
// Created by 小乌嘎 on 2024/2/19.
//

#ifndef FUZZYNONRIGID_EFF_KMEANS_H
#define FUZZYNONRIGID_EFF_KMEANS_H

#include <igl/randperm.h>
#include <igl/repmat.h>
#include <igl/find.h>
#include <igl/min.h>
#include <unsupported/Eigen/KroneckerProduct>

//void repmat(const Eigen::MatrixXd& mat, int n, int m,
//            Eigen::MatrixXd& res) {
//    res = Eigen::KroneckerProduct<Eigen::MatrixXd, Eigen::MatrixXd>(Eigen::MatrixXd::Ones(n, m),
//                                                                    mat);
//}

void sqdist2(Eigen::MatrixXd& aa, Eigen::MatrixXd& a, Eigen::MatrixXd& b,
            Eigen::MatrixXd& res) {
    int a_col = a.cols(), b_col = b.cols();
    Eigen::MatrixXd bb = (b.array() * b.array()).colwise().sum();
    Eigen::MatrixXd ab = a.transpose() * b;
    Eigen::MatrixXd tmp1, tmp2;
    Eigen::MatrixXd aat = aa.transpose();
    igl::repmat(aat, 1, b_col, tmp1);
    igl::repmat(bb, a_col, 1, tmp2);
    res = (tmp1 + tmp2 - 2 * ab).array().abs();
}

void sqdist(Eigen::MatrixXd& a,Eigen::MatrixXd& b,
            Eigen::MatrixXd& res) {
    Eigen::MatrixXd at = a.transpose();
    Eigen::MatrixXd bt = b.transpose();
    Eigen::MatrixXd a2 = (at.array() * at.array()).colwise().sum();
    Eigen::MatrixXd b2 = (bt.array() * bt.array()).colwise().sum();
    Eigen::MatrixXd ab = a * bt;
    Eigen::MatrixXd tmp1, tmp2;
    igl::repmat(a2, 1, b2.cols(), tmp1);
    igl::repmat(b2, a2.cols(), 1, tmp2);
    res = (tmp1 + tmp2 - 2 * ab).array().abs();
}

void eff_kmeans(Eigen::MatrixXd& data, int max_iter,
                Eigen::VectorXi& idx, Eigen::MatrixXd& center, int& m) {
    size_t num = data.rows(), dim = data.cols();
    Eigen::VectorXi dex;
    igl::randperm(num, dex);
    center = data(dex.segment(0, m), Eigen::all);
    for(int i = 0; i < max_iter; i++) {
        Eigen::VectorXd nul = Eigen::VectorXd::Zero(m);
        Eigen::MatrixXd tmp;
        sqdist(center, data, tmp);
        Eigen::VectorXd minn;
        igl::min(tmp, 1, minn, idx);
        for(int j = 0; j < m; j++) {
            igl::find((idx.array() == j).eval(), dex);
            double len = dex.size();
            Eigen::MatrixXd cltr = data(dex, Eigen::all);
            if(len > 1) {
                center(j, Eigen::all) = cltr.colwise().mean();
            }
            else if(len == 1) {
                center(j, Eigen::all) = cltr;
            }
            else {
                nul[j] = 1;
            }
        }
        igl::find((nul.array() == 0).eval(), dex);
        m = dex.size();
        center = center(dex, Eigen::all);
    }
}

#endif //FUZZYNONRIGID_EFF_KMEANS_H
