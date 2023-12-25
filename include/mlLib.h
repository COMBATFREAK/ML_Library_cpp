#pragma once
#include <iostream>
#include <vector>
#include <cassert>
#include <cstdint>

namespace vecAlg
{
    template <typename T>
    std::vector<T> vectorAdd(const std::vector<T> &vectorA, const std::vector<T> &vectorB);
}

namespace matAlg
{
    template <typename T>
    std::vector<std::vector<T>> matrixTranspose(const std::vector<std::vector<T>> &matrixA);

    template <typename T>
    std::vector<std::vector<T>> matrixMultiplication(const std::vector<std::vector<T>> &matrixA, const std::vector<std::vector<T>> &matrixB);
}
