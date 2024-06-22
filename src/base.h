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

#ifndef DEFILLET_IO_H
#define DEFILLET_IO_H
#include "happly.h"
#include <Eigen/Core>
#include <Eigen/Dense>
#include <omp.h>


namespace IO {

    inline void read_3D_points(const std::string &file_path,
                               Eigen::MatrixXd& points) {
        happly::PLYData ply_in(file_path);
        std::vector<double> xPos = ply_in.getElement("vertex").getProperty<double>("x");
        std::vector<double> yPos = ply_in.getElement("vertex").getProperty<double>("y");
        std::vector<double> zPos = ply_in.getElement("vertex").getProperty<double>("z");
        int nb_points = xPos.size();
        points.resize(nb_points, 3);
        #pragma omp parallel for
        for(int i = 0; i < nb_points; i++) {
            points(i, 0) = xPos[i];
            points(i, 1) = yPos[i];
            points(i, 2) = zPos[i];
        }
    }
    inline void read_2D_points(const std::string &file_path,
                               Eigen::MatrixXd& points) {
        happly::PLYData ply_in(file_path);
        std::vector<double> xPos = ply_in.getElement("vertex").getProperty<double>("x");
        std::vector<double> yPos = ply_in.getElement("vertex").getProperty<double>("y");
        int nb_points = xPos.size();
        points.resize(nb_points, 2);
        #pragma omp parallel for
        for(int i = 0; i < nb_points; i++) {
            points(i, 0) = xPos[i];
            points(i, 1) = yPos[i];
        }
    }
    inline void write_3D_points(const std::string &file_path, Eigen::MatrixXd& points, bool binary = false) {
        happly::PLYData plyOut;

        std::string vertexName = "vertex";
        size_t nb_points = points.rows();

        // Create the element
        if (!plyOut.hasElement(vertexName)) {
            plyOut.addElement(vertexName, nb_points);
        }
        // De-interleave
        std::vector<double> xPos(nb_points);
        std::vector<double> yPos(nb_points);
        std::vector<double> zPos(nb_points);

        #pragma omp parallel for
        for (int i = 0; i < nb_points; i++) {
            xPos[i] = points(i, 0);
            yPos[i] = points(i, 1);
            zPos[i] = points(i, 2);
        }

//        std::cout << xPos.size() << std::endl;
//        std::cout << yPos.size() << std::endl;
//        std::cout << zPos.size() << std::endl;
        // Store
        plyOut.getElement(vertexName).addProperty<double>("x", xPos);
        plyOut.getElement(vertexName).addProperty<double>("y", yPos);
        plyOut.getElement(vertexName).addProperty<double>("z", zPos);

        plyOut.write(file_path, binary ? happly::DataFormat::Binary : happly::DataFormat::ASCII);
    }

    inline void write_2D_points(const std::string &file_path, Eigen::MatrixXd& points, bool binary = false) {
        happly::PLYData plyOut;

        std::string vertexName = "vertex";
        size_t nb_points = points.rows();

        // Create the element
        if (!plyOut.hasElement(vertexName)) {
            plyOut.addElement(vertexName, nb_points);
        }

        // De-interleave
        std::vector<double> xPos(nb_points);
        std::vector<double> yPos(nb_points);
#pragma omp parallel for
        for (int i = 0; i < nb_points; i++) {
            xPos[i] = points(i, 0);
            yPos[i] = points(i, 1);
        }

        // Store
        plyOut.getElement(vertexName).addProperty<double>("x", xPos);
        plyOut.getElement(vertexName).addProperty<double>("y", yPos);

        plyOut.write(file_path, binary ? happly::DataFormat::Binary : happly::DataFormat::ASCII);
    }

}

namespace Base {
    struct Kernel {
//    Kernel(const std::string& type, double para) : type_(type), para_(para) {}
        std::string type_;
        double para_;

    };
    class PointSet {
    public:
        PointSet(){}
        PointSet(const Eigen::MatrixXd& points, bool is_normalized = true) {
            set_points(points, is_normalized);
        }
        void set_points(const Eigen::MatrixXd& points, bool is_normalized = true) {
            nb_points_ = points.rows();
            dim_ = points.cols();
            if(nb_points_ == 0) {
                std::cout << "PointSet is empty!" << std::endl;
                return;
            }
            points_ = points;
            scale_ = 1.0;
            centroid_.setZero();
            if(is_normalized) {
                centroid_ = points_.colwise().mean();
                points_.array().rowwise() -= centroid_.transpose().array();
                scale_ = std::sqrt(points_.array().square().sum() / nb_points_);
                points_.array() /= scale_;
            }
        }
    public:
        Eigen::MatrixXd points_;
        Eigen::VectorXd centroid_;
        double scale_;
        int nb_points_;
        int dim_;
    };
}


//        #pragma omp parallel for
//            for(int i = 0; i < nb_points_; i++) {
//                points_(i, 0) = points[i][0];
//                points_(i, 1) = points[i][1];
//                points_(i, 2) = points[i][2];
//            }
#endif //DEFILLET_IO_H
