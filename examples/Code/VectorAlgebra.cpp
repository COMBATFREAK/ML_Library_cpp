#include "../../include/mlLib.h"
#include <iostream>
#include <vector>

int main()
{
  std::vector<double> vectorA = {1.0, 2.0, 3.0};
  std::vector<double> vectorB = {4.0, 5.0, 6.0};
  std::vector<std::vector<double>> matrixA = {{1.0, 2.0}, {3.0, 4.0}};

  // Example usage of vector normalization in the stat namespace
  std::vector<long double> normalizedVectorMinMax = stat::Normalize(vectorA, stat::Min_Max);
  std::vector<long double> normalizedVectorZScore = stat::Normalize(vectorA, stat::Z_Score);

  std::cout << "Normalized Vector (Min-Max): ";
  for (const auto &value : normalizedVectorMinMax)
  {
    std::cout << value << " ";
  }
  std::cout << std::endl;

  std::cout << "Normalized Vector (Z-Score): ";
  for (const auto &value : normalizedVectorZScore)
  {
    std::cout << value << " ";
  }
  std::cout << std::endl;

  // Example usage of vector addition in the vecAlg namespace
  std::vector<double> sumVector = vecAlg::vectorAdd(vectorA, vectorB);

  std::cout << "Sum Vector: ";
  for (const auto &value : sumVector)
  {
    std::cout << value << " ";
  }
  std::cout << std::endl;

  // Example usage of vector subtraction in the vecAlg namespace
  std::vector<double> subtractedVector = vecAlg::vectorSubtract(vectorA, vectorB);

  std::cout << "Subtracted Vector: ";
  for (const auto &value : subtractedVector)
  {
    std::cout << value << " ";
  }
  std::cout << std::endl;

  // Example usage of vector magnitude in the vecAlg namespace
  long double magnitudeA = vecAlg::Magnitude(vectorA);
  long double magnitudeB = vecAlg::Magnitude(vectorB);

  std::cout << "Magnitude of Vector A: " << magnitudeA << std::endl;
  std::cout << "Magnitude of Vector B: " << magnitudeB << std::endl;

  // Example usage of vector dot product in the vecAlg namespace
  size_t dotProduct = vecAlg::DotProduct(vectorA, vectorB);

  std::cout << "Dot Product of Vector A and Vector B: " << dotProduct << std::endl;

  // Example usage of vector angle calculation in the vecAlg namespace
  long double angleDegrees = vecAlg::Angle(vectorA, vectorB, vecAlg::Degree);
  long double angleRadians = vecAlg::Angle(vectorA, vectorB, vecAlg::Radians);

  std::cout << "Angle between Vector A and Vector B (Degrees): " << angleDegrees << std::endl;
  std::cout << "Angle between Vector A and Vector B (Radians): " << angleRadians << std::endl;

  // Example usage of vector scalar multiplication in the vecAlg namespace
  std::vector<long double> scalarMultipliedVector = vecAlg::scalarMultiply(vectorA, 2.0);

  std::cout << "Vector A multiplied by scalar 2.0: ";
  for (const auto &value : scalarMultipliedVector)
  {
    std::cout << value << " ";
  }
  std::cout << std::endl;

  // Example usage of vector scalar division in the vecAlg namespace
  std::vector<long double> scalarDividedVector = vecAlg::scalarDivide(vectorA, 2.0);

  std::cout << "Vector A divided by scalar 2.0: ";
  for (const auto &value : scalarDividedVector)
  {
    std::cout << value << " ";
  }
  std::cout << std::endl;

  // Example usage of elementwise vector multiplication in the vecAlg namespace
  std::vector<long double> elementwiseMultipliedVector = vecAlg::elementwiseMultiply(vectorA, vectorB);

  std::cout << "Elementwise Multiplication of Vector A and Vector B: ";
  for (const auto &value : elementwiseMultipliedVector)
  {
    std::cout << value << " ";
  }
  std::cout << std::endl;

  // Example usage of elementwise vector division in the vecAlg namespace
  std::vector<long double> elementwiseDividedVector = vecAlg::elementwiseDivide(vectorA, vectorB);

  std::cout << "Elementwise Division of Vector A and Vector B: ";
  for (const auto &value : elementwiseDividedVector)
  {
    std::cout << value << " ";
  }
  std::cout << std::endl;

  return 0;
}
