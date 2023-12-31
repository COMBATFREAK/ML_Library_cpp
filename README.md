# ML_Library_cpp
# Machine Learning Library (mlLib)

## Overview

mlLib is a C++ library for machine learning that provides implementations of fundamental algorithms and utilities for statistical and vector/matrix operations. The library is organized into different namespaces, each focusing on specific areas of machine learning and mathematical operations.

## Contents

1. [Introduction](#introduction)
2. [Library Structure](#library-structure)
3. [Namespaces](#namespaces)
4. [Probability and Statistics](#probability-and-statistics)
5. [Vector Algebra](#vector-algebra)
6. [Matrix Algebra](#matrix-algebra)
7. [Machine Learning Models](#machine-learning-models)
    - [Linear Regression](#linear-regression)
    - [Logistic Regression](#logistic-regression)
8. [Usage Examples](#usage-examples)
9. [Build and Installation](#build-and-installation)
10. [Contributing](#contributing)
11. [License](#license)

## Introduction

mlLib is a comprehensive C++ library designed to facilitate machine learning research and applications. It covers a wide range of functionalities, including probability and statistics calculations, vector algebra, matrix algebra, and popular machine learning models like linear and logistic regression.

## Library Structure

The library is organized into the following directories:

- **build:** Contains build artifacts.
- **examples:** Provides usage examples for the library.
- **include:** Holds header files (.h) for the library.
- **lib:** Stores compiled library files.
- **src:** Contains source code files (.cpp) for the library.

## Namespaces

The library uses several namespaces to organize its functionalities:

- **prob:** Probability-related functions.
- **stat:** Statistical functions and utilities.
- **vecAlg:** Vector algebra operations.
- **matAlg:** Matrix algebra operations.
- **mlLib:** Machine learning models and related utilities.

## Probability and Statistics

The `prob` and `stat` namespaces offer functions for probability calculations, combinations, permutations, vector normalization, and statistical metrics.

## Vector Algebra

The `vecAlg` namespace provides functions for vector operations, including addition, subtraction, magnitude calculation, dot product, and more.

## Matrix Algebra

The `matAlg` namespace offers functions for matrix operations such as transposition and multiplication.

## Machine Learning Models

### Linear Regression

The `LinearRegressionModel` class in the `mlLib` namespace represents a linear regression model. It includes functions for prediction and evaluation. Two methods for model creation are available: Least Squares and Gradient Descent.

### Logistic Regression

The `LogisticRegressionModel` class handles logistic regression. It includes functions for prediction, evaluation, and model creation using a specified learning rate and number of iterations.

## Usage Examples

The `examples` directory contains sample code snippets demonstrating how to use the library for various functionalities.

## Build and Installation

Refer to the provided build instructions in the `build` directory for compiling the library and running the examples.

## Contributing

If you wish to contribute to the development of mlLib, please follow the guidelines outlined in the `CONTRIBUTING.md` file.

## License

This library is licensed under the MIT License - see the `LICENSE` file for details. Feel free to use, modify, and distribute it in accordance with the terms of the license.
