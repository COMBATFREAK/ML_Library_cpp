#include "../../include/mlLib.h"
#include <iostream>
#include <iomanip>

int main()
{
    int n = 5;
    int k = 3;
    double p = 0.5;
    double lambda = 2.0;

    // Factorial
    std::cout << "Factorial of " << n << ": " << prob::factorial(n) << std::endl;

    // Combinations
    std::cout << "Combinations(" << n << ", " << k << "): " << prob::combinations(n, k) << std::endl;

    // Permutations
    std::cout << "Permutations(" << n << ", " << k << "): " << prob::permutations(n, k) << std::endl;

    // Binomial Probability
    std::cout << std::fixed << std::setprecision(4);
    std::cout << "Binomial Probability(" << n << ", " << k << ", " << p << "): "
              << prob::binomialProbability(n, k, p) << std::endl;

    // Poisson Probability
    std::cout << "Poisson Probability(" << k << ", " << lambda << "): "
              << prob::poissonProbability(k, lambda) << std::endl;

    return 0;
}
