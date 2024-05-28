#include <iostream>
#include <mkl.h>

int main() {
    const int n = 3; // 矩阵的维度
    double A[9] = { 1.0, 0.2, 0.3,
                    0.2, 2.0, 0.5,
                    0.3, 0.5, 3.0 }; // 实对称矩阵
    double w[3];  // 用于存储特征值
    int info;

    // 调用 LAPACKE_dsyev 函数进行特征值分解
    info = LAPACKE_dsyev(LAPACK_ROW_MAJOR, 'V', 'U', n, A, n, w);

    if (info == 0) {
        std::cout << "Eigenvalues are:\n";
        for (int i = 0; i < n; i++) {
            std::cout << w[i] << std::endl;
        }
        std::cout << "Eigenvectors are:\n";
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                std::cout << A[i * n + j] << " ";
            }
            std::cout << std::endl;
        }
    } else {
        std::cerr << "Error: LAPACKE_dsyev returned " << info << std::endl;
    }

    return 0;
}
