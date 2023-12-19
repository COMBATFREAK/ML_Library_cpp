#pragma once
#include <iostream>
#include <vector>
#include <cassert>
#include <cstdint>

namespace mlLib
{

    template <typename T>
    std::vector<std::vector<T>> matrixTranspose(const std::vector<std::vector<T>> &matrixA);

    template <typename T>
    std::vector<std::vector<T>> matrixMultiplication(const std::vector<std::vector<T>> &matrixA, const std::vector<std::vector<T>> &matrixB);

}
