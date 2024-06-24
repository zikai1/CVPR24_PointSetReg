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

#ifndef FUZZYNONRIGID_KMEANS_H
#define FUZZYNONRIGID_KMEANS_H

#include <omp.h>
#include <chrono>
#include <random>


void sqdist_omp(Eigen::MatrixXd& a,Eigen::MatrixXd& b,
                   Eigen::MatrixXd& res) {
    int n = a.rows(), m = b.rows();
//    res.resize(n, m);
    #pragma omp parallel for collapse(2)
    for(int i = 0; i < n; i++) {
        for(int j = 0; j < m; j++) {
            res(i,j) = (a.row(i) - b.row(j)).squaredNorm();
        }
    }
}

void randperm(Eigen::VectorXi& dex, int a, int b) {
    int n = b - a;
    dex.resize(n);
    std::iota(dex.data(), dex.data() + n, a);
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::default_random_engine rng(seed);
    std::shuffle(dex.data(),dex.data()+n, rng);
}

void kmeans(Eigen::MatrixXd& data, int max_iter,
                Eigen::VectorXi& idx, Eigen::MatrixXd& center, int& m) {
    size_t num = data.rows(), dim = data.cols();
    Eigen::VectorXi dex;
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::default_random_engine rng(seed);
    // igl::randperm(num, dex, rng);
    randperm(dex, 0, num);

    center = data(dex.segment(0, m), Eigen::all);
    Eigen::MatrixXd tmp(center.rows(), data.cols());


    for(int i = 0; i < max_iter; i++) {
        Eigen::VectorXd ct = Eigen::VectorXd::Zero(m);
        tmp.resize(center.rows(), data.rows());
        sqdist_omp(center, data, tmp);

        std::vector<double> min_values(tmp.cols(), std::numeric_limits<double>::max());
        idx.resize(tmp.cols());

        #pragma omp parallel for
        for(int j = 0; j < tmp.cols(); j++) {
            for(int k = 0; k < tmp.rows(); k++) {
                if(min_values[j] > tmp(k, j)) {
                    min_values[j] = tmp(k, j);
                    idx[j] = k;
                }
            }
        }

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

    randperm(dex, 0, num);
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
