#ifndef ML_LIB_H
#define ML_LIB_H

#include <iostream>
#include <vector>
#include <cassert>
#include <cstdint>
#include <numeric>
#include <algorithm>
#include <cmath>

namespace prob
{

    double factorial(int n);

    double combinations(int n, int k);

    double permutations(int n, int k);

    double binomialProbability(int n, int k, double p);

    double poissonProbability(int k, double lambda);

} // namespace probability

namespace stat
{
    // Enumeration for different types of vector normalization
    enum NormalizationType
    {
        Min_Max,
        Z_Score,
        L2_Euclidean
    };

    // Enumeration for different types of vector norms
    enum NormType
    {
        Manhattan,
        Euclidean,
        Infinity
    };

    // Function to calculate a vector norm
    template <typename T>
    long double Norm(const std::vector<T> &vectorA, NormType normType);

    // Function to calculate the mean of a vector
    template <typename T>
    long double Mean(const std::vector<T> &vectorA);

    // Function to normalize a vector
    template <typename T>
    std::vector<long double> Normalize(const std::vector<T> &vectorA, NormalizationType type);

} // namespace Statistics

namespace vecAlg
{
    // Enumeration for specifying angle units
    enum Angles
    {
        Degree,
        Radians
    };

    // Function to add two vectors
    template <typename T>
    std::vector<T> vectorAdd(const std::vector<T> &vectorA, const std::vector<T> &vectorB);

    // Function to subtract one vector from another
    template <typename T>
    std::vector<T> vectorSubtract(const std::vector<T> &vectorA, const std::vector<T> &vectorB);

    // Function to calculate the magnitude of a vector
    template <typename T>
    long double Magnitude(const std::vector<T> &vectorA);

    // Function to calculate the dot product of two vectors
    template <typename T>
    size_t DotProduct(const std::vector<T> &vectorA, const std::vector<T> &vectorB);

    // Function to calculate the angle between two vectors
    template <typename T>
    long double Angle(const std::vector<T> &vectorA, const std::vector<T> &vectorB, Angles angleUnit);

    // Function to multiply a vector by a scalar
    template <typename T>
    std::vector<long double> scalarMultiply(const std::vector<T> &vectorA, long double scalar);

    // Function to divide a vector by a scalar
    template <typename T>
    std::vector<long double> scalarDivide(const std::vector<T> &vectorA, long double scalar);

    // Function to perform elementwise multiplication of two vectors
    template <typename T>
    std::vector<long double> elementwiseMultiply(const std::vector<T> &vectorA, const std::vector<T> &vectorB);

    // Function to perform elementwise division of two vectors
    template <typename T>
    std::vector<long double> elementwiseDivide(const std::vector<T> &vectorA, const std::vector<T> &vectorB);
}

namespace matAlg
{
    // Function to transpose a matrix
    template <typename T>
    std::vector<std::vector<T>> matrixTranspose(const std::vector<std::vector<T>> &matrixA);

    // Function to multiply two matrices
    template <typename T>
    std::vector<std::vector<T>> matrixMultiplication(const std::vector<std::vector<T>> &matrixA, const std::vector<std::vector<T>> &matrixB);
}

namespace mlLib
{

    class LinearRegressionModel
    {
    private:
        long double slope;
        long double intercept;
        stat::NormalizationType normalizationType;

    public:
        // Constructors
        LinearRegressionModel();
        LinearRegressionModel(long double slope, long double intercept, stat::NormalizationType normalizationType);

        // Getter functions
        long double getSlope() const;
        long double getIntercept() const;
        stat::NormalizationType getNormalizationType() const;

        // Setter functions
        void setSlope(long double newSlope);
        void setIntercept(long double newIntercept);
        void setNormalizationType(stat::NormalizationType newNormalizationType);

        template <typename T>
        std::vector<long double> predict(const std::vector<T> &xVector);

        template <typename T>
        long double evaluate(const std::vector<T> &actualYValues, const std::vector<long double> &predictedYValues);
    };

    // Function to create a linear regression model using input X and Y vectors
    template <typename T>
    LinearRegressionModel LinearRegressionLeastSquares(const std::vector<T> &xValues, const std::vector<T> &yValues, stat::NormalizationType normalizationType = stat::NormalizationType::Min_Max);

    template <typename T>
    LinearRegressionModel LinearRegressionGradientDescent(const std::vector<T> &xValues, const std::vector<T> &yValues, stat::NormalizationType normalizationType = stat::NormalizationType::Min_Max, const long double learningRate = 0.01, const int numIterations = 1000);

    class LogisticRegressionModel
    {
    private:
        std::vector<long double> coefficients;
        stat::NormalizationType normalizationType;

    public:
        // Constructors
        LogisticRegressionModel();
        LogisticRegressionModel(const std::vector<long double> &coefficients);

        // Getter function
        const std::vector<long double> &getCoefficients() const;

        // Setter function
        void setCoefficients(const std::vector<long double> &newCoefficients);

        template <typename T>
        std::vector<int> predict(const std::vector<std::vector<T>> &xValues, const long double threshold = 0.5);

        template <typename T>
        long double evaluate(const std::vector<T> &actualYValues, const std::vector<int> &predictedClasses);
    };

    template <typename T>
    LogisticRegressionModel LogisticRegression(const std::vector<std::vector<T>> &xValues, const std::vector<T> &yValues, const long double learningRate = 0.01, const int numIterations = 1000);

} // namespace mlLib

#endif // ML_LIB_H