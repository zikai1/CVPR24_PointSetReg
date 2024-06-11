//
// Created by 小乌嘎 on 2024/2/19.
//

#ifndef FUZZYNONRIGID_KMEANS_H
#define FUZZYNONRIGID_KMEANS_H

#include <igl/randperm.h>
#include <igl/repmat.h>
#include <igl/find.h>
#include <igl/min.h>
#include <unsupported/Eigen/KroneckerProduct>
#include <omp.h>
#include <chrono>


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

void sqdist_matrix(Eigen::MatrixXd& a,Eigen::MatrixXd& b,
            Eigen::MatrixXd& res) {
//    auto start = std::chrono::high_resolution_clock::now();
    Eigen::MatrixXd at = a.transpose();
    Eigen::MatrixXd bt = b.transpose();
    Eigen::MatrixXd a2 = (at.array() * at.array()).colwise().sum();
    Eigen::MatrixXd b2 = (bt.array() * bt.array()).colwise().sum();
    Eigen::MatrixXd ab = a * bt;
    Eigen::MatrixXd tmp1, tmp2;
    igl::repmat(a2.transpose(), 1, b2.cols(), tmp1);
    igl::repmat(b2, a2.cols(), 1, tmp2);
    res = (tmp1 + tmp2 - 2 * ab).array().abs();
}

void sqdist_omp(Eigen::MatrixXd& a,Eigen::MatrixXd& b,
                   Eigen::MatrixXd& res) {
    int n = a.rows(), m = b.rows();
    res.resize(n, m);
    #pragma omp parallel for collapse(2)
    for(int i = 0; i < n; i++) {
        for(int j = 0; j < m; j++) {
            res(i,j) = (a.row(i) - b.row(j)).squaredNorm();
        }
    }
}

void kmeans(Eigen::MatrixXd& data, int max_iter,
                Eigen::VectorXi& idx, Eigen::MatrixXd& center, int& m) {
    size_t num = data.rows(), dim = data.cols();
    Eigen::VectorXi dex;
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::default_random_engine rng(seed);
    igl::randperm(num, dex, rng);
//    for(int i = 0; i < m; i++) {
//        std::cout << dex[i] << std::endl;
//    }
    center = data(dex.segment(0, m), Eigen::all);
//    std::cout << max_iter << std::endl;
    for(int i = 0; i < max_iter; i++) {
        Eigen::VectorXd ct = Eigen::VectorXd::Zero(m);
        Eigen::MatrixXd tmp;
        sqdist_omp(center, data, tmp);
        Eigen::VectorXd minn;
        igl::min(tmp, 1, minn, idx);
        center.setZero();
        #pragma omp parallel for
        for(int j = 0; j < num ; j++) {
            #pragma omp critical
            {
                center.row(idx[j]) += data.row(j);
                ct[idx[j]]++;
            }
        }

        #pragma omp parallel for
        for(int j = 0; j < m; j++) {
            center.row(j) /= ct[j];
        }
    }
}


void elkan_kmeans(Eigen::MatrixXd& data, int max_iter,
            Eigen::VectorXi& idx, Eigen::MatrixXd& center, int& m) {
    size_t num = data.rows(), dim = data.cols();
    Eigen::VectorXi dex;
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::default_random_engine rng(seed);
    igl::randperm(num, dex);
    center = data(dex.segment(0, m), Eigen::all);
    Eigen::VectorXd tmp(m);
    idx.resize(num);
    idx.setConstant(-1);
    Eigen::VectorXi clust_size(m); clust_size.setZero();
    Eigen::MatrixXd center_sum(center.rows(), center.cols());
    center_sum.setZero();

    for(int iter = 0; iter < max_iter; iter++) {
        #pragma omp parallel for collapse(2)
        for (int i = 0; i < m; i++) {
            tmp[i] = std::numeric_limits<double>::max();
            for (int j = 0; j < m; j++) {
                if (i == j) continue;
                double dis = (center.row(i) - center.row(j)).squaredNorm();
                if (dis < tmp[i]) {
                    tmp[i] = dis;
                }
            }
        }

//        clust_size.setZero();
        #pragma omp parallel for collapse(2)
        for (int i = 0; i < num; i++) {
            if(idx[i] == -1) {
                double minn = std::numeric_limits<double>::max();
                int id = 0;
                for(int j = 0; j < m; j++) {
                    double dis = (data.row(i) - center.row(j)).squaredNorm();
                    if(dis < minn) {
                        minn = dis; id = j;
                        if(2 * dis < tmp[j]) break;
                    }
                }
                idx[i] = id;
                clust_size[id]++;
                center_sum.row(id) += data.row(i);
            }
            else {
                double dis = (data.row(i) - center.row(idx[i])).squaredNorm();
                if(2 * dis <= tmp[idx[i]]) {
                    continue;
                } else {
                    double minn = std::numeric_limits<double>::max();
                    int id = 0;
                    for(int j = 0; j < m; j++) {
                        dis = (data.row(i) - center.row(j)).squaredNorm();
                        if(dis < minn) {
                            minn = dis; id = j;
                            if(2 * dis <= tmp[j]) break;
                        }
                    }
                    clust_size[idx[i]]--;
                    center_sum.row(idx[i]) -= data.row(i);

                    idx[i] = id;
                    clust_size[idx[i]]++;
                    center_sum.row(idx[i]) += data.row(i);
                }
            }
        }

        #pragma omp parallel for
        for(int i = 0; i < m; i++) {
            center.row(i) = center_sum.row(i) / clust_size[i];
//            std::cout << center_sum.row(i) << ' ' << clust_size[i] << std::endl;
        }
    }
}

#endif //FUZZYNONRIGID_KMEANS_H
