#ifndef ML_LIB_H
#define ML_LIB_H

#include <iostream>
#include <fstream>
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
// mlLib.h
namespace mlLib
{
    // Struct to represent a confusion matrix
    struct ConfusionMatrix
    {
        size_t truePositive = 0;
        size_t trueNegative = 0;
        size_t falsePositive = 0;
        size_t falseNegative = 0;
    };

    // Struct to store evaluation metrics
    struct EvaluationMetrics
    {
        long double accuracy = 0;
        long double recall = 0;
        long double precision = 0;
        long double f1Score = 0;
    };

    inline std::ostream &operator<<(std::ostream &os, const ConfusionMatrix &obj)
    {
        os << obj.truePositive << " "
           << obj.trueNegative << " "
           << obj.falsePositive << " "
           << obj.falseNegative << " ";
        return os;
    }

    // Deserialization function for ConfusionMatrix
    inline std::istream &operator>>(std::istream &is, ConfusionMatrix &obj)
    {
        is >> obj.truePositive >> obj.trueNegative >> obj.falsePositive >> obj.falseNegative;
        return is;
    }

    // Serialization function for EvaluationMetrics
    inline std::ostream &operator<<(std::ostream &os, const EvaluationMetrics &obj)
    {
        os << obj.accuracy << " "
           << obj.recall << " "
           << obj.precision << " "
           << obj.f1Score << " ";
        return os;
    }

    // Deserialization function for EvaluationMetrics
    inline std::istream &operator>>(std::istream &is, EvaluationMetrics &obj)
    {
        is >> obj.accuracy >> obj.recall >> obj.precision >> obj.f1Score;
        return is;
    }

    // Class for Linear Regression Model
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

        // Predict function
        template <typename T>
        std::vector<long double> predict(const std::vector<T> &xVector);

        // Evaluate function
        template <typename T>
        long double evaluate(const std::vector<T> &actualYValues, const std::vector<long double> &predictedYValues);

        void saveToFile(const std::string &filename);
        void loadFromFile(const std::string &filename);

        // Serialization function
        friend std::ostream &operator<<(std::ostream &os, const LinearRegressionModel &obj)
        {
            os << obj.slope << " " << obj.intercept << " " << static_cast<int>(obj.normalizationType);
            return os;
        }

        // Deserialization function
        friend std::istream &operator>>(std::istream &is, LinearRegressionModel &obj)
        {
            int normalizationTypeInt;
            is >> obj.slope >> obj.intercept >> normalizationTypeInt;
            obj.normalizationType = static_cast<stat::NormalizationType>(normalizationTypeInt);
            return is;
        }

        void printInfo() const;
    };

    // Function to create a linear regression model using Least Squares method
    template <typename T>
    LinearRegressionModel LinearRegressionLeastSquares(const std::vector<T> &xValues, const std::vector<T> &yValues, stat::NormalizationType normalizationType = stat::NormalizationType::Min_Max);

    // Function to create a linear regression model using Gradient Descent method
    template <typename T>
    LinearRegressionModel LinearRegressionGradientDescent(const std::vector<T> &xValues, const std::vector<T> &yValues, stat::NormalizationType normalizationType = stat::NormalizationType::Min_Max, const long double learningRate = 0.01, const int numIterations = 1000);

    // Class for Logistic Regression Model
    class LogisticRegressionModel
    {
    private:
        std::vector<long double> coefficients;
        stat::NormalizationType normalizationType;
        ConfusionMatrix confusionMatrix;
        EvaluationMetrics evaluationMetrics;

    public:
        // Constructors
        LogisticRegressionModel();
        LogisticRegressionModel(const std::vector<long double> &coefficients);

        // Getter functions
        const std::vector<long double> &getCoefficients() const;
        ConfusionMatrix getConfusionMatrix() const;
        EvaluationMetrics getEvaluationMetrics() const;

        // Setter functions
        void setCoefficients(const std::vector<long double> &newCoefficients);
        void setConfusionMatrix(const ConfusionMatrix &matrix);
        void setEvaluationMetrics(const EvaluationMetrics &metrics);

        // Predict function
        template <typename T>
        std::vector<int> predict(const std::vector<std::vector<T>> &xValues, const long double threshold = 0.5);

        // Evaluate function
        template <typename T>
        long double evaluate(const std::vector<T> &actualYValues, const std::vector<int> &predictedClasses);

        // Save the object to a file
        void saveToFile(const std::string &filename) const;

        // Load the object from a file
        void loadFromFile(const std::string &filename);

        friend std::ostream &operator<<(std::ostream &os, const LogisticRegressionModel &obj)
        {
            // Serialize coefficients
            os << obj.coefficients.size() << " ";
            for (const auto &coef : obj.coefficients)
            {
                os << coef << " ";
            }

            // Serialize other members
            os << static_cast<int>(obj.normalizationType) << " ";

            // Serialize ConfusionMatrix
            os << obj.confusionMatrix;

            // Serialize EvaluationMetrics
            os << obj.evaluationMetrics;

            return os;
        }

        // Definition of the deserialization function
        friend std::istream &operator>>(std::istream &is, LogisticRegressionModel &obj)
        {
            // Deserialize coefficients
            size_t coefSize;
            is >> coefSize;
            obj.coefficients.resize(coefSize);
            for (size_t i = 0; i < coefSize; ++i)
            {
                is >> obj.coefficients[i];
            }

            // Deserialize other members
            int normalizationTypeInt;
            is >> normalizationTypeInt;
            obj.normalizationType = static_cast<stat::NormalizationType>(normalizationTypeInt);

            // Deserialize ConfusionMatrix
            is >> obj.confusionMatrix;

            // Deserialize EvaluationMetrics
            is >> obj.evaluationMetrics;

            return is;
        }

        void printInfo() const;
    };

    // Function to create a logistic regression model
    template <typename T>
    LogisticRegressionModel LogisticRegression(const std::vector<std::vector<T>> &xValues, const std::vector<T> &yValues, const long double learningRate = 0.01, const int numIterations = 1000);

} // namespace mlLib

#endif // ML_LIB_H