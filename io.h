//
// Created by xiaowuga on 12/17/2023.
//

#ifndef DEFILLET_IO_H
#define DEFILLET_IO_H
#include "happly.h"

namespace IO {

    inline void read_ply_mesh(const std::string &file_path, std::vector<std::array<double, 3>> &points,
                              std::vector<std::vector<size_t>> &faces) {
        happly::PLYData ply_in(file_path);
        points = ply_in.getVertexPositions();
        faces = ply_in.getFaceIndices<size_t>();
    }

    inline void write_ply_mesh(const std::string &file_path, std::vector<std::array<double, 3>> &points,
                               std::vector<std::vector<size_t>> &faces,
                               bool binary = false) {
        happly::PLYData plyOut;
        plyOut.addVertexPositions(points);
        plyOut.addFaceIndices(faces);
        plyOut.write(file_path, binary ? happly::DataFormat::Binary : happly::DataFormat::ASCII);
    }

    inline void read_ply_points(const std::string &file_path, std::vector<std::array<double, 3>> &points) {
        happly::PLYData ply_in(file_path);
        points = ply_in.getVertexPositions();
    }

    inline void
    write_ply_points(const std::string &file_path, std::vector<std::array<double, 3>> &points, bool binary = false) {
        happly::PLYData plyOut;
        plyOut.addVertexPositions(points);
        plyOut.write(file_path, binary ? happly::DataFormat::Binary : happly::DataFormat::ASCII);
    }

}

#endif //DEFILLET_IO_H
