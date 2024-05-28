#include "MatlabEngine.hpp"
#include "MatlabDataArray.hpp"
#include <iostream>
using namespace matlab::engine;
using namespace matlab::data;
std::unique_ptr<MATLABEngine> matlabPtr = startMATLAB();
int main() {


    auto start = std::chrono::high_resolution_clock::now();
    // 启动MATLAB引擎

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    std::cout << "time: " << duration.count() << " s" << std::endl;

    // 创建一个3x3矩阵
    ArrayFactory factory;
    TypedArray<double> matrix = factory.createArray<double>({3, 3},
                                                            {1.0, 2.0, 3.0,
                                                             4.0, 5.0, 6.0,
                                                             7.0, 8.0, 9.0});

    // 将矩阵传递给MATLAB并计算特征值和特征向量
    matlabPtr->setVariable(u"matrix", matrix);
    matlabPtr->eval(u"[V, D] = eig(matrix);");

    // 获取特征值和特征向量
    TypedArray<double> eigenValues = matlabPtr->getVariable(u"D");
    TypedArray<double> eigenVectors = matlabPtr->getVariable(u"V");

    // 打印特征值
    std::cout << "Eigenvalues:" << std::endl;
    for (auto val : eigenValues) {
        std::cout << val << " ";
    }
    std::cout << std::endl;

    // 打印特征向量
    std::cout << "Eigenvectors:" << std::endl;
    for (auto row : eigenVectors) {
        std::cout << row << std::endl;
    }

    return 0;
}
