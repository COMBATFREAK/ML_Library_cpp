#include "../include/mlLib.h"

namespace prob
{

    double factorial(int n)
    {
        if (n == 0 || n == 1)
            return 1;

        double result = 1;
        for (int i = 2; i <= n; ++i)
        {
            result *= i;
        }

        return result;
    }

    double combinations(int n, int k)
    {
        if (k > n)
            return 0;
        return prob::factorial(n) / (prob::factorial(k) * prob::factorial(n - k));
    }

    double permutations(int n, int k)
    {
        if (k > n)
            return 0;
        return prob::factorial(n) / prob::factorial(n - k);
    }

    double binomialProbability(int n, int k, double p)
    {
        if (k > n || p < 0 || p > 1)
            return 0.0;
        return prob::combinations(n, k) * std::pow(p, k) * std::pow(1 - p, n - k);
    }

    double poissonProbability(int k, double lambda)
    {
        if (lambda < 0)
            return 0.0;
        return (std::exp(-lambda) * std::pow(lambda, k)) / prob::factorial(k);
    }

} // namespace probability

namespace stat
{
    template <typename T>
    long double Norm(const std::vector<T> &vectorA, NormType normType)
    {
        assert(("Vector is empty" && !vectorA.empty()));

        const size_t vecLen = vectorA.size();
        long double result = 0.0L;

        switch (normType)
        {
        case Manhattan:
            for (size_t i = 0; i < vecLen; ++i)
            {
                result += std::abs(static_cast<long double>(vectorA[i]));
            }
            break;

        case Euclidean:
            for (size_t i = 0; i < vecLen; ++i)
            {
                result += std::pow(static_cast<long double>(vectorA[i]), 2);
            }
            result = std::sqrt(result);
            break;

        case Infinity:
            for (size_t i = 0; i < vecLen; ++i)
            {
                result = std::max(result, std::abs(static_cast<long double>(vectorA[i])));
            }
            break;

        default:
            assert(false && "Unsupported norm type");
        }

        return result;
    }

    template long double Norm(const std::vector<int8_t> &vectorA, NormType normType);
    template long double Norm(const std::vector<int16_t> &vectorA, NormType normType);
    template long double Norm(const std::vector<int32_t> &vectorA, NormType normType);
    template long double Norm(const std::vector<int64_t> &vectorA, NormType normType);

    template long double Norm(const std::vector<uint8_t> &vectorA, NormType normType);
    template long double Norm(const std::vector<uint16_t> &vectorA, NormType normType);
    template long double Norm(const std::vector<uint32_t> &vectorA, NormType normType);
    template long double Norm(const std::vector<uint64_t> &vectorA, NormType normType);

    template long double Norm(const std::vector<float> &vectorA, NormType normType);
    template long double Norm(const std::vector<double> &vectorA, NormType normType);
    template long double Norm(const std::vector<long double> &vectorA, NormType normType);

    template <typename T>
    long double Mean(const std::vector<T> &vectorA)
    {
        assert(("Vector is empty" && !vectorA.empty()));

        return static_cast<long double>(std::accumulate(vectorA.begin(), vectorA.end(), 0)) / vectorA.size();
    }

    template long double Mean(const std::vector<int8_t> &vectorA);
    template long double Mean(const std::vector<int16_t> &vectorA);
    template long double Mean(const std::vector<int32_t> &vectorA);
    template long double Mean(const std::vector<int64_t> &vectorA);

    template long double Mean(const std::vector<uint8_t> &vectorA);
    template long double Mean(const std::vector<uint16_t> &vectorA);
    template long double Mean(const std::vector<uint32_t> &vectorA);
    template long double Mean(const std::vector<uint64_t> &vectorA);

    template long double Mean(const std::vector<float> &vectorA);
    template long double Mean(const std::vector<double> &vectorA);
    template long double Mean(const std::vector<long double> &vectorA);

    template <typename T>
    long double standardDeviation(const std::vector<T> &vectorA)
    {
        assert(vectorA.size() >= 2 && "Vector has insufficient elements for standard deviation");

        const long double meanValue = Mean(vectorA);
        long double sumSquaredDiffs = 0.0L;

        size_t vecLen = vectorA.size();
        for (size_t i = 0; i < vecLen; ++i)
        {
            long double diff = static_cast<long double>(vectorA[i]) - meanValue;
            sumSquaredDiffs += (diff * diff);
        }

        return std::sqrt(sumSquaredDiffs / static_cast<long double>(vecLen - 1));
    }

    template long double standardDeviation(const std::vector<int8_t> &vectorA);
    template long double standardDeviation(const std::vector<int16_t> &vectorA);
    template long double standardDeviation(const std::vector<int32_t> &vectorA);
    template long double standardDeviation(const std::vector<int64_t> &vectorA);

    template long double standardDeviation(const std::vector<uint8_t> &vectorA);
    template long double standardDeviation(const std::vector<uint16_t> &vectorA);
    template long double standardDeviation(const std::vector<uint32_t> &vectorA);
    template long double standardDeviation(const std::vector<uint64_t> &vectorA);

    template long double standardDeviation(const std::vector<float> &vectorA);
    template long double standardDeviation(const std::vector<double> &vectorA);
    template long double standardDeviation(const std::vector<long double> &vectorA);

    template <typename T>
    std::vector<long double> Normalize(const std::vector<T> &vectorA, NormalizationType type)
    {
        assert(("Vector is empty" && !vectorA.empty()));

        const size_t vecLen = vectorA.size();
        std::vector<long double> normalizedVector(vecLen, 0);

        switch (type)
        {
        case Min_Max:
        {
            const auto maxIter = std::max_element(vectorA.begin(), vectorA.end());
            const auto minIter = std::min_element(vectorA.begin(), vectorA.end());

            const auto maxValue = static_cast<long double>(*maxIter);
            const auto minValue = static_cast<long double>(*minIter);

            for (size_t i = 0; i < vecLen; i++)
            {
                normalizedVector[i] = (static_cast<long double>(vectorA[i]) - minValue) / (maxValue - minValue);
            }
        }
        break;

        case Z_Score:
        {
            const long double vecMean = Mean(vectorA);
            const long double vecStdDev = standardDeviation(vectorA);

            for (size_t i = 0; i < vecLen; i++)
            {
                normalizedVector[i] = (static_cast<long double>(vectorA[i]) - vecMean) / vecStdDev;
            }
        }
        break;

        case L2_Euclidean:
        {
            const long double euclideanNorm = Norm(vectorA, Euclidean);

            for (size_t i = 0; i < vecLen; i++)
            {
                normalizedVector[i] = static_cast<long double>(vectorA[i]) / euclideanNorm;
            }
        }
        break;

        default:
            assert(false && "Unsupported norm type");
            break;
        }

        return normalizedVector;
    }

    template std::vector<long double> Normalize(const std::vector<int8_t> &vectorA, NormalizationType type);
    template std::vector<long double> Normalize(const std::vector<int16_t> &vectorA, NormalizationType type);
    template std::vector<long double> Normalize(const std::vector<int32_t> &vectorA, NormalizationType type);
    template std::vector<long double> Normalize(const std::vector<int64_t> &vectorA, NormalizationType type);

    template std::vector<long double> Normalize(const std::vector<uint8_t> &vectorA, NormalizationType type);
    template std::vector<long double> Normalize(const std::vector<uint16_t> &vectorA, NormalizationType type);
    template std::vector<long double> Normalize(const std::vector<uint32_t> &vectorA, NormalizationType type);
    template std::vector<long double> Normalize(const std::vector<uint64_t> &vectorA, NormalizationType type);

    template std::vector<long double> Normalize(const std::vector<float> &vectorA, NormalizationType type);
    template std::vector<long double> Normalize(const std::vector<double> &vectorA, NormalizationType type);
    template std::vector<long double> Normalize(const std::vector<long double> &vectorA, NormalizationType type);

} // namespace Statistics

namespace vecAlg
{
    template <typename T>
    std::vector<T> vectorAdd(const std::vector<T> &vectorA, const std::vector<T> &vectorB)
    {
        assert(("Vector 1 is empty" && !vectorA.empty()));
        assert(("Vector 2 is empty" && !vectorB.empty()));
        assert(("Vector dimensions do not match" && vectorA.size() == vectorB.size()));

        const size_t vecLen = vectorA.size();
        std::vector<T> result(vecLen, 0);

        for (size_t i = 0; i < vecLen; i++)
        {
            result[i] = vectorA[i] + vectorB[i];
        }

        return result;
    }

    template std::vector<int8_t> vectorAdd(const std::vector<int8_t> &vectorA, const std::vector<int8_t> &vectorB);
    template std::vector<int16_t> vectorAdd(const std::vector<int16_t> &vectorA, const std::vector<int16_t> &vectorB);
    template std::vector<int32_t> vectorAdd(const std::vector<int32_t> &vectorA, const std::vector<int32_t> &vectorB);
    template std::vector<int64_t> vectorAdd(const std::vector<int64_t> &vectorA, const std::vector<int64_t> &vectorB);

    template std::vector<uint8_t> vectorAdd(const std::vector<uint8_t> &vectorA, const std::vector<uint8_t> &vectorB);
    template std::vector<uint16_t> vectorAdd(const std::vector<uint16_t> &vectorA, const std::vector<uint16_t> &vectorB);
    template std::vector<uint32_t> vectorAdd(const std::vector<uint32_t> &vectorA, const std::vector<uint32_t> &vectorB);
    template std::vector<uint64_t> vectorAdd(const std::vector<uint64_t> &vectorA, const std::vector<uint64_t> &vectorB);

    template std::vector<float> vectorAdd(const std::vector<float> &vectorA, const std::vector<float> &vectorB);
    template std::vector<double> vectorAdd(const std::vector<double> &vectorA, const std::vector<double> &vectorB);
    template std::vector<long double> vectorAdd(const std::vector<long double> &vectorA, const std::vector<long double> &vectorB);

    template <typename T>
    std::vector<T> vectorSubtract(const std::vector<T> &vectorA, const std::vector<T> &vectorB)
    {
        assert(("Vector 1 is empty" && !vectorA.empty()) &&
               ("Vector 2 is empty" && !vectorB.empty()) &&
               ("Vector dimensions do not match" && vectorA.size() == vectorB.size()));

        const size_t vecLen = vectorA.size();
        std::vector<T> result(vecLen, 0);

        for (size_t i = 0; i < vecLen; i++)
        {
            result[i] = vectorA[i] - vectorB[i];
        }

        return result;
    }

    template std::vector<int8_t> vectorSubtract(const std::vector<int8_t> &vectorA, const std::vector<int8_t> &vectorB);
    template std::vector<int16_t> vectorSubtract(const std::vector<int16_t> &vectorA, const std::vector<int16_t> &vectorB);
    template std::vector<int32_t> vectorSubtract(const std::vector<int32_t> &vectorA, const std::vector<int32_t> &vectorB);
    template std::vector<int64_t> vectorSubtract(const std::vector<int64_t> &vectorA, const std::vector<int64_t> &vectorB);

    template std::vector<uint8_t> vectorSubtract(const std::vector<uint8_t> &vectorA, const std::vector<uint8_t> &vectorB);
    template std::vector<uint16_t> vectorSubtract(const std::vector<uint16_t> &vectorA, const std::vector<uint16_t> &vectorB);
    template std::vector<uint32_t> vectorSubtract(const std::vector<uint32_t> &vectorA, const std::vector<uint32_t> &vectorB);
    template std::vector<uint64_t> vectorSubtract(const std::vector<uint64_t> &vectorA, const std::vector<uint64_t> &vectorB);

    template std::vector<float> vectorSubtract(const std::vector<float> &vectorA, const std::vector<float> &vectorB);
    template std::vector<double> vectorSubtract(const std::vector<double> &vectorA, const std::vector<double> &vectorB);
    template std::vector<long double> vectorSubtract(const std::vector<long double> &vectorA, const std::vector<long double> &vectorB);

    template <typename T>
    long double Magnitude(const std::vector<T> &vectorA)
    {
        assert(("Vector is empty" && !vectorA.empty()));

        const size_t vecLen = vectorA.size();
        long double result = 0;

        for (size_t i = 0; i < vecLen; i++)
        {
            result += static_cast<long double>(vectorA[i]) * static_cast<long double>(vectorA[i]);
        }

        return std::sqrt(result);
    }

    template long double Magnitude(const std::vector<int8_t> &vectorA);
    template long double Magnitude(const std::vector<int16_t> &vectorA);
    template long double Magnitude(const std::vector<int32_t> &vectorA);
    template long double Magnitude(const std::vector<int64_t> &vectorA);

    template long double Magnitude(const std::vector<uint8_t> &vectorA);
    template long double Magnitude(const std::vector<uint16_t> &vectorA);
    template long double Magnitude(const std::vector<uint32_t> &vectorA);
    template long double Magnitude(const std::vector<uint64_t> &vectorA);

    template long double Magnitude(const std::vector<float> &vectorA);
    template long double Magnitude(const std::vector<double> &vectorA);
    template long double Magnitude(const std::vector<long double> &vectorA);

    template <typename T>
    size_t DotProduct(const std::vector<T> &vectorA, const std::vector<T> &vectorB)
    {
        assert(("Vector 1 is empty" && !vectorA.empty()) &&
               ("Vector 2 is empty" && !vectorB.empty()) &&
               ("Vector dimensions do not match" && vectorA.size() == vectorB.size()));

        const size_t vecLen = vectorA.size();
        size_t result = 0;

        for (size_t i = 0; i < vecLen; i++)
        {
            result += vectorA[i] * vectorB[i];
        }

        return result;
    }

    template size_t DotProduct(const std::vector<int8_t> &vectorA, const std::vector<int8_t> &vectorB);
    template size_t DotProduct(const std::vector<int16_t> &vectorA, const std::vector<int16_t> &vectorB);
    template size_t DotProduct(const std::vector<int32_t> &vectorA, const std::vector<int32_t> &vectorB);
    template size_t DotProduct(const std::vector<int64_t> &vectorA, const std::vector<int64_t> &vectorB);

    template size_t DotProduct(const std::vector<uint8_t> &vectorA, const std::vector<uint8_t> &vectorB);
    template size_t DotProduct(const std::vector<uint16_t> &vectorA, const std::vector<uint16_t> &vectorB);
    template size_t DotProduct(const std::vector<uint32_t> &vectorA, const std::vector<uint32_t> &vectorB);
    template size_t DotProduct(const std::vector<uint64_t> &vectorA, const std::vector<uint64_t> &vectorB);

    template size_t DotProduct(const std::vector<float> &vectorA, const std::vector<float> &vectorB);
    template size_t DotProduct(const std::vector<double> &vectorA, const std::vector<double> &vectorB);
    template size_t DotProduct(const std::vector<long double> &vectorA, const std::vector<long double> &vectorB);

    template <typename T>
    long double Angle(const std::vector<T> &vectorA, const std::vector<T> &vectorB, Angles angleUnit)
    {
        assert(("Vector 1 is empty" && !vectorA.empty()) &&
               ("Vector 2 is empty" && !vectorB.empty()) &&
               ("Vector dimensions do not match" && vectorA.size() == vectorB.size()));

        const long double dotProd = DotProduct(vectorA, vectorB);
        const long double magnitudeA = Magnitude(vectorA);
        const long double magnitudeB = Magnitude(vectorB);

        long double cos_theta = dotProd / (magnitudeA * magnitudeB);

        cos_theta = std::max(-1.0L, std::min(1.0L, cos_theta));
        long double theta = std::acos(cos_theta);

        if (angleUnit == Degree)
        {
            theta = theta * 180.0 / M_PI;
        }

        return theta;
    }

    template long double vecAlg::Angle(const std::vector<int8_t> &vectorA, const std::vector<int8_t> &vectorB, Angles angleUnit);
    template long double vecAlg::Angle(const std::vector<int16_t> &vectorA, const std::vector<int16_t> &vectorB, Angles angleUnit);
    template long double vecAlg::Angle(const std::vector<int32_t> &vectorA, const std::vector<int32_t> &vectorB, Angles angleUnit);
    template long double vecAlg::Angle(const std::vector<int64_t> &vectorA, const std::vector<int64_t> &vectorB, Angles angleUnit);

    template long double vecAlg::Angle(const std::vector<uint8_t> &vectorA, const std::vector<uint8_t> &vectorB, Angles angleUnit);
    template long double vecAlg::Angle(const std::vector<uint16_t> &vectorA, const std::vector<uint16_t> &vectorB, Angles angleUnit);
    template long double vecAlg::Angle(const std::vector<uint32_t> &vectorA, const std::vector<uint32_t> &vectorB, Angles angleUnit);
    template long double vecAlg::Angle(const std::vector<uint64_t> &vectorA, const std::vector<uint64_t> &vectorB, Angles angleUnit);

    template long double vecAlg::Angle(const std::vector<float> &vectorA, const std::vector<float> &vectorB, Angles angleUnit);
    template long double vecAlg::Angle(const std::vector<double> &vectorA, const std::vector<double> &vectorB, Angles angleUnit);
    template long double vecAlg::Angle(const std::vector<long double> &vectorA, const std::vector<long double> &vectorB, Angles angleUnit);

    template <typename T>
    std::vector<long double> scalarMultiply(const std::vector<T> &vectorA, long double scalar)
    {

        assert(("Vector is empty" && !vectorA.empty()));

        const size_t vecLen = vectorA.size();
        std::vector<long double> result(vecLen, 0);

        if (scalar == 0)
            return result;

        for (size_t i = 0; i < vecLen; i++)
        {
            result[i] = vectorA[i] * scalar;
        }

        return result;
    }
    template std::vector<long double> vecAlg::scalarMultiply(const std::vector<int8_t> &vectorA, long double scalar);
    template std::vector<long double> vecAlg::scalarMultiply(const std::vector<int16_t> &vectorA, long double scalar);
    template std::vector<long double> vecAlg::scalarMultiply(const std::vector<int32_t> &vectorA, long double scalar);
    template std::vector<long double> vecAlg::scalarMultiply(const std::vector<int64_t> &vectorA, long double scalar);

    template std::vector<long double> vecAlg::scalarMultiply(const std::vector<uint8_t> &vectorA, long double scalar);
    template std::vector<long double> vecAlg::scalarMultiply(const std::vector<uint16_t> &vectorA, long double scalar);
    template std::vector<long double> vecAlg::scalarMultiply(const std::vector<uint32_t> &vectorA, long double scalar);
    template std::vector<long double> vecAlg::scalarMultiply(const std::vector<uint64_t> &vectorA, long double scalar);

    template std::vector<long double> vecAlg::scalarMultiply(const std::vector<float> &vectorA, long double scalar);
    template std::vector<long double> vecAlg::scalarMultiply(const std::vector<double> &vectorA, long double scalar);
    template std::vector<long double> vecAlg::scalarMultiply(const std::vector<long double> &vectorA, long double scalar);

    template <typename T>
    std::vector<long double> scalarDivide(const std::vector<T> &vectorA, long double scalar)
    {
        assert((("Vector is empty" && !vectorA.empty()) &&
                ("Scalar Divison by 0 Not possible" && scalar != 0)));

        const size_t vecLen = vectorA.size();
        std::vector<long double> result(vecLen, 0);

        for (size_t i = 0; i < vecLen; i++)
        {
            result[i] = vectorA[i] / scalar;
        }

        return result;
    }

    template std::vector<long double> vecAlg::scalarDivide(const std::vector<int8_t> &vectorA, long double scalar);
    template std::vector<long double> vecAlg::scalarDivide(const std::vector<int16_t> &vectorA, long double scalar);
    template std::vector<long double> vecAlg::scalarDivide(const std::vector<int32_t> &vectorA, long double scalar);
    template std::vector<long double> vecAlg::scalarDivide(const std::vector<int64_t> &vectorA, long double scalar);

    template std::vector<long double> vecAlg::scalarDivide(const std::vector<uint8_t> &vectorA, long double scalar);
    template std::vector<long double> vecAlg::scalarDivide(const std::vector<uint16_t> &vectorA, long double scalar);
    template std::vector<long double> vecAlg::scalarDivide(const std::vector<uint32_t> &vectorA, long double scalar);
    template std::vector<long double> vecAlg::scalarDivide(const std::vector<uint64_t> &vectorA, long double scalar);

    template std::vector<long double> vecAlg::scalarDivide(const std::vector<float> &vectorA, long double scalar);
    template std::vector<long double> vecAlg::scalarDivide(const std::vector<double> &vectorA, long double scalar);
    template std::vector<long double> vecAlg::scalarDivide(const std::vector<long double> &vectorA, long double scalar);

    template <typename T>
    std::vector<long double> elementwiseMultiply(const std::vector<T> &vectorA, const std::vector<T> &vectorB)
    {
        assert(("Vector 1 is empty" && !vectorA.empty()) &&
               ("Vector 2 is empty" && !vectorB.empty()) &&
               ("Vector dimensions do not match" && vectorA.size() == vectorB.size()));

        const size_t vecLen = vectorA.size();
        std::vector<long double> result(vecLen, 0);

        for (size_t i = 0; i < vecLen; i++)
        {
            result[i] = static_cast<long double>(vectorA[i]) * static_cast<long double>(vectorB[i]);
        }

        return result;
    }

    template std::vector<long double> vecAlg::elementwiseMultiply(const std::vector<int8_t> &vectorA, const std::vector<int8_t> &vectorB);
    template std::vector<long double> vecAlg::elementwiseMultiply(const std::vector<int16_t> &vectorA, const std::vector<int16_t> &vectorB);
    template std::vector<long double> vecAlg::elementwiseMultiply(const std::vector<int32_t> &vectorA, const std::vector<int32_t> &vectorB);
    template std::vector<long double> vecAlg::elementwiseMultiply(const std::vector<int64_t> &vectorA, const std::vector<int64_t> &vectorB);

    template std::vector<long double> vecAlg::elementwiseMultiply(const std::vector<uint8_t> &vectorA, const std::vector<uint8_t> &vectorB);
    template std::vector<long double> vecAlg::elementwiseMultiply(const std::vector<uint16_t> &vectorA, const std::vector<uint16_t> &vectorB);
    template std::vector<long double> vecAlg::elementwiseMultiply(const std::vector<uint32_t> &vectorA, const std::vector<uint32_t> &vectorB);
    template std::vector<long double> vecAlg::elementwiseMultiply(const std::vector<uint64_t> &vectorA, const std::vector<uint64_t> &vectorB);

    template std::vector<long double> vecAlg::elementwiseMultiply(const std::vector<float> &vectorA, const std::vector<float> &vectorB);
    template std::vector<long double> vecAlg::elementwiseMultiply(const std::vector<double> &vectorA, const std::vector<double> &vectorB);
    template std::vector<long double> vecAlg::elementwiseMultiply(const std::vector<long double> &vectorA, const std::vector<long double> &vectorB);

    template <typename T>
    std::vector<long double> elementwiseDivide(const std::vector<T> &vectorA, const std::vector<T> &vectorB)
    {

        assert(("Vector 1 is empty" && !vectorA.empty()) &&
               ("Vector 2 is empty" && !vectorB.empty()) &&
               ("Vector dimensions do not match" && vectorA.size() == vectorB.size()));

        const size_t vecLen = vectorA.size();
        std::vector<long double> result(vecLen, 0);

        for (size_t i = 0; i < vecLen; i++)
        {
            assert(("Error: Division by zero." && vectorB[i] != 0));

            result[i] = static_cast<long double>(vectorA[i]) / static_cast<long double>(vectorB[i]);
        }

        return result;
    }

    template std::vector<long double> vecAlg::elementwiseDivide(const std::vector<int8_t> &vectorA, const std::vector<int8_t> &vectorB);
    template std::vector<long double> vecAlg::elementwiseDivide(const std::vector<int16_t> &vectorA, const std::vector<int16_t> &vectorB);
    template std::vector<long double> vecAlg::elementwiseDivide(const std::vector<int32_t> &vectorA, const std::vector<int32_t> &vectorB);
    template std::vector<long double> vecAlg::elementwiseDivide(const std::vector<int64_t> &vectorA, const std::vector<int64_t> &vectorB);

    template std::vector<long double> vecAlg::elementwiseDivide(const std::vector<uint8_t> &vectorA, const std::vector<uint8_t> &vectorB);
    template std::vector<long double> vecAlg::elementwiseDivide(const std::vector<uint16_t> &vectorA, const std::vector<uint16_t> &vectorB);
    template std::vector<long double> vecAlg::elementwiseDivide(const std::vector<uint32_t> &vectorA, const std::vector<uint32_t> &vectorB);
    template std::vector<long double> vecAlg::elementwiseDivide(const std::vector<uint64_t> &vectorA, const std::vector<uint64_t> &vectorB);

    template std::vector<long double> vecAlg::elementwiseDivide(const std::vector<float> &vectorA, const std::vector<float> &vectorB);
    template std::vector<long double> vecAlg::elementwiseDivide(const std::vector<double> &vectorA, const std::vector<double> &vectorB);
    template std::vector<long double> vecAlg::elementwiseDivide(const std::vector<long double> &vectorA, const std::vector<long double> &vectorB);

} // namespace vecAlg

namespace matAlg
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

        const size_t rowsA = matrixA.size();
        const size_t colsA = matrixA[0].size();
        const size_t colsB = matrixB[0].size();

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

    // Template instantiation for various types
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

} // namespace matAlg

namespace mlLib
{
    LinearRegressionModel::LinearRegressionModel(){};

    LinearRegressionModel::LinearRegressionModel(long double slope, long double intercept, stat::NormalizationType normalizationType)
        : slope(slope), intercept(intercept), normalizationType(normalizationType) {}

    // Getter function implementations
    long double LinearRegressionModel::getSlope() const { return slope; }
    long double LinearRegressionModel::getIntercept() const { return intercept; }
    stat::NormalizationType LinearRegressionModel::getNormalizationType() const { return normalizationType; }

    // Setter function implementations
    void LinearRegressionModel::setSlope(long double newSlope) { slope = newSlope; }
    void LinearRegressionModel::setIntercept(long double newIntercept) { intercept = newIntercept; }
    void LinearRegressionModel::setNormalizationType(stat::NormalizationType newNormalizationType) { normalizationType = newNormalizationType; }

    template <typename T>
    std::vector<long double> LinearRegressionModel::predict(const std::vector<T> &xValues)
    {
        assert(("xValues is empty" && !xValues.empty()) && "Input values must not be empty");

        size_t size = xValues.size();

        std::vector<long double> normalizedXValues = stat::Normalize(xValues, normalizationType);
        std::vector<long double> predictedYValues(size);

        for (size_t i = 0; i < size; ++i)
        {
            predictedYValues[i] = slope * normalizedXValues[i] + intercept;
        }

        return predictedYValues;
    }

    template std::vector<long double> LinearRegressionModel::predict(const std::vector<int8_t> &xValues);
    template std::vector<long double> LinearRegressionModel::predict(const std::vector<int16_t> &xValues);
    template std::vector<long double> LinearRegressionModel::predict(const std::vector<int32_t> &xValues);
    template std::vector<long double> LinearRegressionModel::predict(const std::vector<int64_t> &xValues);

    template std::vector<long double> LinearRegressionModel::predict(const std::vector<uint8_t> &xValues);
    template std::vector<long double> LinearRegressionModel::predict(const std::vector<uint16_t> &xValues);
    template std::vector<long double> LinearRegressionModel::predict(const std::vector<uint32_t> &xValues);
    template std::vector<long double> LinearRegressionModel::predict(const std::vector<uint64_t> &xValues);

    template std::vector<long double> LinearRegressionModel::predict(const std::vector<float> &xValues);
    template std::vector<long double> LinearRegressionModel::predict(const std::vector<double> &xValues);
    template std::vector<long double> LinearRegressionModel::predict(const std::vector<long double> &xValues);

    template <typename T>
    long double LinearRegressionModel::evaluate(const std::vector<T> &actualYValues, const std::vector<long double> &predictedYValues)
    {
        assert(("Actual Y values are empty" && !actualYValues.empty()) &&
               ("Predicted Y values are empty" && !predictedYValues.empty()) &&
               (actualYValues.size() == predictedYValues.size() && "Input values must have the same size"));

        size_t size = actualYValues.size();
        long double mse = 0.0L;

        // Normalize actualYValues based on the model's normalization type
        std::vector<long double> normalizedActualYValues = stat::Normalize(actualYValues, normalizationType);

        for (size_t i = 0; i < size; ++i)
        {
            mse += std::pow(normalizedActualYValues[i] - predictedYValues[i], 2);
        }

        mse /= static_cast<long double>(size);
        return mse;
    }

    template long double LinearRegressionModel::evaluate(const std::vector<int8_t> &actualYValues, const std::vector<long double> &predictedYValues);
    template long double LinearRegressionModel::evaluate(const std::vector<int16_t> &actualYValues, const std::vector<long double> &predictedYValues);
    template long double LinearRegressionModel::evaluate(const std::vector<int32_t> &actualYValues, const std::vector<long double> &predictedYValues);
    template long double LinearRegressionModel::evaluate(const std::vector<int64_t> &actualYValues, const std::vector<long double> &predictedYValues);

    template long double LinearRegressionModel::evaluate(const std::vector<uint8_t> &actualYValues, const std::vector<long double> &predictedYValues);
    template long double LinearRegressionModel::evaluate(const std::vector<uint16_t> &actualYValues, const std::vector<long double> &predictedYValues);
    template long double LinearRegressionModel::evaluate(const std::vector<uint32_t> &actualYValues, const std::vector<long double> &predictedYValues);
    template long double LinearRegressionModel::evaluate(const std::vector<uint64_t> &actualYValues, const std::vector<long double> &predictedYValues);

    template long double LinearRegressionModel::evaluate(const std::vector<float> &actualYValues, const std::vector<long double> &predictedYValues);
    template long double LinearRegressionModel::evaluate(const std::vector<double> &actualYValues, const std::vector<long double> &predictedYValues);
    template long double LinearRegressionModel::evaluate(const std::vector<long double> &actualYValues, const std::vector<long double> &predictedYValues);

    void LinearRegressionModel::saveToFile(const std::string &filename)
    {
        std::ofstream outFile(filename);
        if (outFile.is_open())
        {
            outFile << *this;
            outFile.close();
        }
        else
        {
            std::cerr << "Unable to open file for writing." << std::endl;
        }
    }

    void LinearRegressionModel::loadFromFile(const std::string &filename)
    {
        std::ifstream inFile(filename);
        if (inFile.is_open())
        {
            inFile >> *this;
            inFile.close();
        }
        else
        {
            std::cerr << "Unable to open file for reading." << std::endl;
        }
    }

    template <typename T>
    LinearRegressionModel LinearRegressionLeastSquares(const std::vector<T> &xValues, const std::vector<T> &yValues, stat::NormalizationType normalizationType)
    {
        assert(("xValues is empty" && !xValues.empty()) &&
               ("yValues is empty" && !yValues.empty()) &&
               (xValues.size() == yValues.size() && "Input vectors must have the same size"));

        std::vector<long double> normalizedXValues = stat::Normalize(xValues, normalizationType);
        std::vector<long double> normalizedYValues = stat::Normalize(yValues, normalizationType);

        size_t size = normalizedXValues.size();

        long double sumX = std::accumulate(normalizedXValues.begin(), normalizedXValues.end(), 0.0L);
        long double sumY = std::accumulate(normalizedYValues.begin(), normalizedYValues.end(), 0.0L);
        long double sumXY = vecAlg::DotProduct(normalizedXValues, normalizedYValues);
        long double sumXSquare = vecAlg::DotProduct(normalizedXValues, normalizedXValues);

        long double slope = (size * sumXY - sumX * sumY) / (size * sumXSquare - std::pow(sumX, 2));
        long double intercept = (sumY - slope * sumX) / size;

        return LinearRegressionModel(slope, intercept, normalizationType);
    }

    void LinearRegressionModel::printInfo() const
    {
        std::cout << "Linear Regression Model:\n"
                  << "Slope: " << slope << "\n"
                  << "Intercept: " << intercept << "\n"
                  << "Normalization Type: " << normalizationType << "\n";
    }

    template LinearRegressionModel LinearRegressionLeastSquares(const std::vector<int8_t> &xValues, const std::vector<int8_t> &yValues, stat::NormalizationType normalizationType);
    template LinearRegressionModel LinearRegressionLeastSquares(const std::vector<int16_t> &xValues, const std::vector<int16_t> &yValues, stat::NormalizationType normalizationType);
    template LinearRegressionModel LinearRegressionLeastSquares(const std::vector<int32_t> &xValues, const std::vector<int32_t> &yValues, stat::NormalizationType normalizationType);
    template LinearRegressionModel LinearRegressionLeastSquares(const std::vector<int64_t> &xValues, const std::vector<int64_t> &yValues, stat::NormalizationType normalizationType);

    template LinearRegressionModel LinearRegressionLeastSquares(const std::vector<uint8_t> &xValues, const std::vector<uint8_t> &yValues, stat::NormalizationType normalizationType);
    template LinearRegressionModel LinearRegressionLeastSquares(const std::vector<uint16_t> &xValues, const std::vector<uint16_t> &yValues, stat::NormalizationType normalizationType);
    template LinearRegressionModel LinearRegressionLeastSquares(const std::vector<uint32_t> &xValues, const std::vector<uint32_t> &yValues, stat::NormalizationType normalizationType);
    template LinearRegressionModel LinearRegressionLeastSquares(const std::vector<uint64_t> &xValues, const std::vector<uint64_t> &yValues, stat::NormalizationType normalizationType);

    template LinearRegressionModel LinearRegressionLeastSquares(const std::vector<float> &xValues, const std::vector<float> &yValues, stat::NormalizationType normalizationType);
    template LinearRegressionModel LinearRegressionLeastSquares(const std::vector<double> &xValues, const std::vector<double> &yValues, stat::NormalizationType normalizationType);
    template LinearRegressionModel LinearRegressionLeastSquares(const std::vector<long double> &xValues, const std::vector<long double> &yValues, stat::NormalizationType normalizationType);

    template <typename T>
    LinearRegressionModel LinearRegressionGradientDescent(const std::vector<T> &xValues, const std::vector<T> &yValues, stat::NormalizationType normalizationType, long double learningRate, int numIterations)
    {
        assert(("xValues is empty" && !xValues.empty()) &&
               ("yValues is empty" && !yValues.empty()) &&
               (xValues.size() == yValues.size() && "Input vectors must have the same size"));

        std::vector<long double> normalizedXValues = stat::Normalize(xValues, normalizationType);
        std::vector<long double> normalizedYValues = stat::Normalize(yValues, normalizationType);

        size_t size = normalizedXValues.size();

        // Initialize parameters
        long double slope = 0.0L;
        long double intercept = 0.0L;

        // Gradient Descent
        for (int iteration = 0; iteration < numIterations; ++iteration)
        {
            long double sumErrors = 0.0L;
            long double sumXErrors = 0.0L;

            for (size_t i = 0; i < size; ++i)
            {
                long double error = slope * normalizedXValues[i] + intercept - normalizedYValues[i];
                sumErrors += error;
                sumXErrors += error * normalizedXValues[i];
            }

            // Update parameters
            slope -= learningRate * (1.0L / size) * sumXErrors;
            intercept -= learningRate * (1.0L / size) * sumErrors;
        }

        return LinearRegressionModel(slope, intercept, normalizationType);
    }

    template LinearRegressionModel LinearRegressionGradientDescent(const std::vector<int8_t> &xValues, const std::vector<int8_t> &yValues, stat::NormalizationType normalizationType, const long double learningRate, const int numIterations);
    template LinearRegressionModel LinearRegressionGradientDescent(const std::vector<int16_t> &xValues, const std::vector<int16_t> &yValues, stat::NormalizationType normalizationType, const long double learningRate, const int numIterations);
    template LinearRegressionModel LinearRegressionGradientDescent(const std::vector<int32_t> &xValues, const std::vector<int32_t> &yValues, stat::NormalizationType normalizationType, const long double learningRate, const int numIterations);
    template LinearRegressionModel LinearRegressionGradientDescent(const std::vector<int64_t> &xValues, const std::vector<int64_t> &yValues, stat::NormalizationType normalizationType, const long double learningRate, const int numIterations);

    template LinearRegressionModel LinearRegressionGradientDescent(const std::vector<uint8_t> &xValues, const std::vector<uint8_t> &yValues, stat::NormalizationType normalizationType, const long double learningRate, const int numIterations);
    template LinearRegressionModel LinearRegressionGradientDescent(const std::vector<uint16_t> &xValues, const std::vector<uint16_t> &yValues, stat::NormalizationType normalizationType, const long double learningRate, const int numIterations);
    template LinearRegressionModel LinearRegressionGradientDescent(const std::vector<uint32_t> &xValues, const std::vector<uint32_t> &yValues, stat::NormalizationType normalizationType, const long double learningRate, const int numIterations);
    template LinearRegressionModel LinearRegressionGradientDescent(const std::vector<uint64_t> &xValues, const std::vector<uint64_t> &yValues, stat::NormalizationType normalizationType, const long double learningRate, const int numIterations);

    template LinearRegressionModel LinearRegressionGradientDescent(const std::vector<float> &xValues, const std::vector<float> &yValues, stat::NormalizationType normalizationType, const long double learningRate, const int numIterations);
    template LinearRegressionModel LinearRegressionGradientDescent(const std::vector<double> &xValues, const std::vector<double> &yValues, stat::NormalizationType normalizationType, const long double learningRate, const int numIterations);
    template LinearRegressionModel LinearRegressionGradientDescent(const std::vector<long double> &xValues, const std::vector<long double> &yValues, stat::NormalizationType normalizationType, const long double learningRate, const int numIterations);

    LogisticRegressionModel::LogisticRegressionModel() {}
    LogisticRegressionModel::LogisticRegressionModel(const std::vector<long double> &coefficients) : coefficients(coefficients), normalizationType(stat::NormalizationType::Z_Score) {}

    // Getter function
    const std::vector<long double> &LogisticRegressionModel::getCoefficients() const { return coefficients; }

    ConfusionMatrix LogisticRegressionModel::getConfusionMatrix() const { return confusionMatrix; }

    // Getter and Setter for Evaluation Metrics
    EvaluationMetrics LogisticRegressionModel::getEvaluationMetrics() const { return evaluationMetrics; }

    // Setter function
    void LogisticRegressionModel::setCoefficients(const std::vector<long double> &newCoefficients) { coefficients = newCoefficients; }

    void LogisticRegressionModel::setConfusionMatrix(const ConfusionMatrix &matrix) { confusionMatrix = matrix; }

    void LogisticRegressionModel::setEvaluationMetrics(const EvaluationMetrics &metrics) { evaluationMetrics = metrics; }

    template <typename T>
    std::vector<int> LogisticRegressionModel::predict(const std::vector<std::vector<T>> &xValues, const long double threshold)
    {
        assert(!coefficients.empty() && "Model coefficients are not initialized.");

        std::vector<int> predictions;
        predictions.reserve(xValues.size());

        for (size_t i = 0; i < xValues.size(); ++i)
        {
            const auto &xVector = xValues[i];
            assert(xVector.size() == coefficients.size() - 1 && "Input feature size mismatch.");

            long double logit = coefficients[0];
            for (size_t j = 0; j < xVector.size(); ++j)
            {
                logit += coefficients[j + 1] * xVector[j];
            }

            const long double probability = 1.0 / (1.0 + std::exp(-logit));
            const int predictedClass = (probability >= threshold) ? 1 : 0;
            predictions.push_back(predictedClass);
        }

        return predictions;
    }

    template std::vector<int> LogisticRegressionModel::predict(const std::vector<std::vector<int8_t>> &xValues, long double threshold);
    template std::vector<int> LogisticRegressionModel::predict(const std::vector<std::vector<int16_t>> &xValues, long double threshold);
    template std::vector<int> LogisticRegressionModel::predict(const std::vector<std::vector<int32_t>> &xValues, long double threshold);
    template std::vector<int> LogisticRegressionModel::predict(const std::vector<std::vector<int64_t>> &xValues, long double threshold);

    template std::vector<int> LogisticRegressionModel::predict(const std::vector<std::vector<uint8_t>> &xValues, long double threshold);
    template std::vector<int> LogisticRegressionModel::predict(const std::vector<std::vector<uint16_t>> &xValues, long double threshold);
    template std::vector<int> LogisticRegressionModel::predict(const std::vector<std::vector<uint32_t>> &xValues, long double threshold);
    template std::vector<int> LogisticRegressionModel::predict(const std::vector<std::vector<uint64_t>> &xValues, long double threshold);

    template std::vector<int> LogisticRegressionModel::predict(const std::vector<std::vector<float>> &xValues, long double threshold);
    template std::vector<int> LogisticRegressionModel::predict(const std::vector<std::vector<double>> &xValues, long double threshold);
    template std::vector<int> LogisticRegressionModel::predict(const std::vector<std::vector<long double>> &xValues, long double threshold);

    template <typename T>
    long double LogisticRegressionModel::evaluate(const std::vector<T> &actualYValues, const std::vector<int> &predictedClasses)
    {
        assert(("Input vectors must have the same size" && actualYValues.size() == predictedClasses.size()));

        size_t dataSize = actualYValues.size();
        ConfusionMatrix newConfusionMatrix = {0, 0, 0, 0}; // Initialize to zeros

        for (size_t i = 0; i < dataSize; ++i)
        {
            int predictedClass = predictedClasses[i];
            int actualValue = actualYValues[i];

            if (predictedClass == 1 && actualValue == 1)
            {
                newConfusionMatrix.truePositive++;
            }
            else if (predictedClass == 0 && actualValue == 0)
            {
                newConfusionMatrix.trueNegative++;
            }
            else if (predictedClass == 1 && actualValue == 0)
            {
                newConfusionMatrix.falsePositive++;
            }
            else if (predictedClass == 0 && actualValue == 1)
            {
                newConfusionMatrix.falseNegative++;
            }
        }

        setConfusionMatrix(newConfusionMatrix);

        long double accuracy = static_cast<long double>(newConfusionMatrix.truePositive + newConfusionMatrix.trueNegative) / dataSize * 100.0;
        long double recall = static_cast<long double>(newConfusionMatrix.truePositive) / (newConfusionMatrix.truePositive + newConfusionMatrix.falseNegative) * 100.0;
        long double precision = static_cast<long double>(newConfusionMatrix.truePositive) / (newConfusionMatrix.truePositive + newConfusionMatrix.falsePositive) * 100.0;
        long double f1Score = 2 * precision * recall / (precision + recall);

        // Set the evaluation metrics
        setEvaluationMetrics({accuracy, recall, precision, f1Score});

        size_t correctPredictions = newConfusionMatrix.truePositive + newConfusionMatrix.trueNegative;
        return static_cast<long double>(correctPredictions) / dataSize * 100.0;
    }

    template long double mlLib::LogisticRegressionModel::evaluate<int8_t>(const std::vector<int8_t> &actualYValues, const std::vector<int> &predictedClasses);
    template long double mlLib::LogisticRegressionModel::evaluate<int16_t>(const std::vector<int16_t> &actualYValues, const std::vector<int> &predictedClasses);
    template long double mlLib::LogisticRegressionModel::evaluate<int32_t>(const std::vector<int32_t> &actualYValues, const std::vector<int> &predictedClasses);
    template long double mlLib::LogisticRegressionModel::evaluate<int64_t>(const std::vector<int64_t> &actualYValues, const std::vector<int> &predictedClasses);

    template long double mlLib::LogisticRegressionModel::evaluate<uint8_t>(const std::vector<uint8_t> &actualYValues, const std::vector<int> &predictedClasses);
    template long double mlLib::LogisticRegressionModel::evaluate<uint16_t>(const std::vector<uint16_t> &actualYValues, const std::vector<int> &predictedClasses);
    template long double mlLib::LogisticRegressionModel::evaluate<uint32_t>(const std::vector<uint32_t> &actualYValues, const std::vector<int> &predictedClasses);
    template long double mlLib::LogisticRegressionModel::evaluate<uint64_t>(const std::vector<uint64_t> &actualYValues, const std::vector<int> &predictedClasses);

    template long double mlLib::LogisticRegressionModel::evaluate<float>(const std::vector<float> &actualYValues, const std::vector<int> &predictedClasses);
    template long double mlLib::LogisticRegressionModel::evaluate<double>(const std::vector<double> &actualYValues, const std::vector<int> &predictedClasses);
    template long double mlLib::LogisticRegressionModel::evaluate<long double>(const std::vector<long double> &actualYValues, const std::vector<int> &predictedClasses);

    void LogisticRegressionModel::saveToFile(const std::string &filename) const
    {
        std::ofstream outFile(filename);
        if (outFile.is_open())
        {
            outFile << *this;
            outFile.close();
        }
        else
        {
            std::cerr << "Unable to open file for writing." << std::endl;
        }
    }

    // Definition of the member function loadFromFile
    void LogisticRegressionModel::loadFromFile(const std::string &filename)
    {
        std::ifstream inFile(filename);
        if (inFile.is_open())
        {
            inFile >> *this;
            inFile.close();
        }
        else
        {
            std::cerr << "Unable to open file for reading." << std::endl;
        }
    }

    void LogisticRegressionModel::printInfo() const
    {
        std::cout << "Logistic Regression Model:\n"
                  << "Coefficients: ";
        for (const auto &coeff : coefficients)
        {
            std::cout << coeff << " ";
        }
        std::cout << "\nNormalization Type: " << normalizationType << "\n"
                  << "Confusion Matrix: TP=" << confusionMatrix.truePositive
                  << ", TN=" << confusionMatrix.trueNegative
                  << ", FP=" << confusionMatrix.falsePositive
                  << ", FN=" << confusionMatrix.falseNegative << "\n"
                  << "Evaluation Metrics: Accuracy=" << evaluationMetrics.accuracy
                  << ", Recall=" << evaluationMetrics.recall
                  << ", Precision=" << evaluationMetrics.precision
                  << ", F1 Score=" << evaluationMetrics.f1Score << "\n";
    }

    template <typename T>
    LogisticRegressionModel LogisticRegression(const std::vector<std::vector<T>> &xValues, const std::vector<T> &yValues, const long double learningRate, const int numIterations)
    {
        assert(!xValues.empty() && "xValues is empty");
        assert(!yValues.empty() && "yValues is empty");
        assert(xValues.size() == yValues.size() && "Input vectors must have the same size");

        size_t numFeatures = xValues[0].size();
        std::vector<long double> coefficients(numFeatures + 1, 0.0);

        for (int iter = 0; iter < numIterations; ++iter)
        {
            std::vector<long double> errors(yValues.size(), 0.0);
            long double interceptGradient = 0.0;

            for (size_t i = 0; i < xValues.size(); ++i)
            {
                long double logit = coefficients[0];
                for (size_t j = 0; j < numFeatures; ++j)
                {
                    logit += coefficients[j + 1] * xValues[i][j];
                }

                const long double prediction = 1.0 / (1.0 + std::exp(-logit));
                errors[i] = prediction - yValues[i];

                interceptGradient += errors[i];
                for (size_t j = 0; j < numFeatures; ++j)
                {
                    coefficients[j + 1] -= learningRate * errors[i] * xValues[i][j];
                }
            }

            interceptGradient /= yValues.size();
            coefficients[0] -= learningRate * interceptGradient;
        }

        return LogisticRegressionModel(coefficients);
    }

    template LogisticRegressionModel LogisticRegression(const std::vector<std::vector<int8_t>> &xValues, const std::vector<int8_t> &yValues, const long double learningRate, const int numIterations);
    template LogisticRegressionModel LogisticRegression(const std::vector<std::vector<int16_t>> &xValues, const std::vector<int16_t> &yValues, const long double learningRate, const int numIterations);
    template LogisticRegressionModel LogisticRegression(const std::vector<std::vector<int32_t>> &xValues, const std::vector<int32_t> &yValues, const long double learningRate, const int numIterations);
    template LogisticRegressionModel LogisticRegression(const std::vector<std::vector<int64_t>> &xValues, const std::vector<int64_t> &yValues, const long double learningRate, const int numIterations);

    // Explicit instantiations for Logistic Regression with uint types
    template LogisticRegressionModel LogisticRegression(const std::vector<std::vector<uint8_t>> &xValues, const std::vector<uint8_t> &yValues, const long double learningRate, const int numIterations);
    template LogisticRegressionModel LogisticRegression(const std::vector<std::vector<uint16_t>> &xValues, const std::vector<uint16_t> &yValues, const long double learningRate, const int numIterations);
    template LogisticRegressionModel LogisticRegression(const std::vector<std::vector<uint32_t>> &xValues, const std::vector<uint32_t> &yValues, const long double learningRate, const int numIterations);
    template LogisticRegressionModel LogisticRegression(const std::vector<std::vector<uint64_t>> &xValues, const std::vector<uint64_t> &yValues, const long double learningRate, const int numIterations);

    // Explicit instantiations for Logistic Regression with float types
    template LogisticRegressionModel LogisticRegression(const std::vector<std::vector<float>> &xValues, const std::vector<float> &yValues, const long double learningRate, const int numIterations);
    template LogisticRegressionModel LogisticRegression(const std::vector<std::vector<double>> &xValues, const std::vector<double> &yValues, const long double learningRate, const int numIterations);
    template LogisticRegressionModel LogisticRegression(const std::vector<std::vector<long double>> &xValues, const std::vector<long double> &yValues, const long double learningRate, const int numIterations);

} // namespace mlLib
