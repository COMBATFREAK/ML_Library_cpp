#include "../include/mlLib.h"

namespace mlLib
{
    template <typename T>
    std::vector<std::vector<T>> matrixTranspose(const std::vector<std::vector<T>> &matrixA)
    {

        assert(("Matrix is empty" && !matrixA.empty()) &&
               ("Matrix has empty rows" && !matrixA[0].empty()));

        std::vector<std::vector<T>> transposeMatrix(matrixA[0].size(), std::vector<T>(matrixA.size(), 0));
        for (size_t i = 0; i < matrixA.size(); ++i)
        {
            for (size_t j = 0; j < matrixA[0].size(); ++j)
            {
                transposeMatrix[j][i] = matrixA[i][j];
            }
        }

        return transposeMatrix;
    }

    template std::vector<std::vector<int8_t>> matrixTranspose(const std::vector<std::vector<int8_t>> &matrixA);
    template std::vector<std::vector<int16_t>> matrixTranspose(const std::vector<std::vector<int16_t>> &matrixA);
    template std::vector<std::vector<int32_t>> matrixTranspose(const std::vector<std::vector<int32_t>> &matrixA);
    template std::vector<std::vector<int64_t>> matrixTranspose(const std::vector<std::vector<int64_t>> &matrixA);

    template std::vector<std::vector<uint8_t>> matrixTranspose(const std::vector<std::vector<uint8_t>> &matrixA);
    template std::vector<std::vector<uint16_t>> matrixTranspose(const std::vector<std::vector<uint16_t>> &matrixA);
    template std::vector<std::vector<uint32_t>> matrixTranspose(const std::vector<std::vector<uint32_t>> &matrixA);
    template std::vector<std::vector<uint64_t>> matrixTranspose(const std::vector<std::vector<uint64_t>> &matrixA);

    template std::vector<std::vector<float>> matrixTranspose(const std::vector<std::vector<float>> &matrixA);
    template std::vector<std::vector<double>> matrixTranspose(const std::vector<std::vector<double>> &matrixA);
    template std::vector<std::vector<long double>> matrixTranspose(const std::vector<std::vector<long double>> &matrixA);

    template <typename T>
    std::vector<std::vector<T>> matrixMultiplication(const std::vector<std::vector<T>> &matrixA, const std::vector<std::vector<T>> &matrixB)
    {

        assert(("Matrix 1 is empty" && !matrixA.empty()) &&
               ("Matrix 2 is empty" && !matrixB.empty()) &&
               ("Matrix 1 has empty rows" && !matrixA[0].empty()) &&
               ("Matrix 2 has empty rows" && !matrixB[0].empty()) &&
               ("Matrix dimensions do not match" && matrixA[0].size() == matrixB.size()));

        std::vector<std::vector<T>> transposeMatrixB = matrixTranspose(matrixB);

        std::vector<std::vector<T>> result(matrixA.size(), std::vector<T>(matrixB[0].size(), 0));

        size_t rowsA = matrixA.size();
        size_t colsA = matrixA[0].size();
        size_t colsB = matrixB[0].size();

        for (size_t i = 0; i < rowsA; ++i)
        {
            for (size_t j = 0; j < colsB; ++j)
            {
                for (size_t k = 0; k < colsA; ++k)
                {
                    result[i][j] += matrixA[i][k] * transposeMatrixB[j][k];
                }
            }
        }

        return result;
    }
    template std::vector<std::vector<int8_t>> matrixMultiplication(const std::vector<std::vector<int8_t>> &matrixA, const std::vector<std::vector<int8_t>> &matrixB);
    template std::vector<std::vector<int16_t>> matrixMultiplication(const std::vector<std::vector<int16_t>> &matrixA, const std::vector<std::vector<int16_t>> &matrixB);
    template std::vector<std::vector<int32_t>> matrixMultiplication(const std::vector<std::vector<int32_t>> &matrixA, const std::vector<std::vector<int32_t>> &matrixB);
    template std::vector<std::vector<int64_t>> matrixMultiplication(const std::vector<std::vector<int64_t>> &matrixA, const std::vector<std::vector<int64_t>> &matrixB);

    template std::vector<std::vector<uint8_t>> matrixMultiplication(const std::vector<std::vector<uint8_t>> &matrixA, const std::vector<std::vector<uint8_t>> &matrixB);
    template std::vector<std::vector<uint16_t>> matrixMultiplication(const std::vector<std::vector<uint16_t>> &matrixA, const std::vector<std::vector<uint16_t>> &matrixB);
    template std::vector<std::vector<uint32_t>> matrixMultiplication(const std::vector<std::vector<uint32_t>> &matrixA, const std::vector<std::vector<uint32_t>> &matrixB);
    template std::vector<std::vector<uint64_t>> matrixMultiplication(const std::vector<std::vector<uint64_t>> &matrixA, const std::vector<std::vector<uint64_t>> &matrixB);

    template std::vector<std::vector<float>> matrixMultiplication(const std::vector<std::vector<float>> &matrixA, const std::vector<std::vector<float>> &matrixB);
    template std::vector<std::vector<double>> matrixMultiplication(const std::vector<std::vector<double>> &matrixA, const std::vector<std::vector<double>> &matrixB);
    template std::vector<std::vector<long double>> matrixMultiplication(const std::vector<std::vector<long double>> &matrixA, const std::vector<std::vector<long double>> &matrixB);

} // namespace mlLib
