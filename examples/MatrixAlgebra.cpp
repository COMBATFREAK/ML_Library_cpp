#include "../include/mlLib.h"
#include <iostream>
#include <vector>

int main()
{
    // Example matrix A
    std::vector<std::vector<double>> matrixA = {{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}};

    // Transpose matrix A
    std::vector<std::vector<double>> transposedMatrix = matAlg::matrixTranspose(matrixA);

    std::cout << "Transposed Matrix:" << std::endl;
    for (const auto &row : transposedMatrix)
    {
        for (const auto &value : row)
        {
            std::cout << value << " ";
        }
        std::cout << std::endl;
    }

    // Example matrix B
    std::vector<std::vector<double>> matrixB = {{5.0, 6.0}, {7.0, 8.0}};

    // Multiply matrices A and B
    std::vector<std::vector<double>> multipliedMatrix = matAlg::matrixMultiplication(transposedMatrix, matrixB);

    std::cout << "Multiplied Matrix:" << std::endl;
    for (const auto &row : multipliedMatrix)
    {
        for (const auto &value : row)
        {
            std::cout << value << " ";
        }
        std::cout << std::endl;
    }

    return 0;
}
