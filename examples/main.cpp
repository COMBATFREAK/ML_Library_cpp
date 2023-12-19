#include "../include/mlLib.h"
#include <iostream>
#include <vector>

int main()
{

    std::vector<std::vector<int>> matrixA(1000, std::vector<int>(1000, 1));
    std::vector<std::vector<int>> matrixB(1000, std::vector<int>(1000, 2));

    auto resultMatrix = mlLib::matrixMultiplication(matrixA, matrixB);

    return 0;
}
