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
    - [Compilation Commands](#compilation-commands)
10. [Contributing](#contributing)
11. [License](#license)

## Introduction

mlLib is a comprehensive C++ library designed to facilitate machine learning research and applications. It covers a wide range of functionalities, including probability and statistics calculations, vector algebra, matrix algebra, and popular machine learning models like linear and logistic regression.

## Library Structure

The library is organized into the following directories:

- **build:** Contains build artifacts.
- **examples:** Provides usage examples for the library.
  - **Code:** Contains example codes.
  - **DataSet:** Holds datasets for linear and logistic regression.
  - **Exe:** Stores the compiled executables of example codes.
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

The `examples` directory contains subdirectories:

- **Code:** Contains sample code snippets demonstrating how to use the library for various functionalities.
- **DataSet:** Holds datasets for linear and logistic regression.
- **Exe:** Stores the compiled executables of example codes.

## Build and Installation

### Compilation Commands

Use the following commands to compile the library and example codes:

```bash
# Compile mlLib.cpp source file into an object file
g++ -c src/mlLib.cpp -o build/mlLib.o

# Create a static library (archive) containing the mlLib object file
ar rcs lib/mlLib.a build/mlLib.o

# Compile examples
g++ examples/Code/VectorAlgebra.cpp -Iinclude/ -Llib/ -l:mlLib.a -o examples/Exe/t1
g++ examples/Code/MatrixAlgebra.cpp -Iinclude/ -Llib/ -l:mlLib.a -o examples/Exe/t2
g++ examples/Code/Probability.cpp -Iinclude/ -Llib/ -l:mlLib.a -o examples/Exe/t3
g++ examples/Code/LinearRegressionLeastSquares.cpp -Iinclude/ -Llib/ -l:mlLib.a -o examples/Exe/t4
g++ examples/Code/LinearRegressionGradientDescent.cpp -Iinclude/ -Llib/ -l:mlLib.a -o examples/Exe/t5
g++ examples/Code/LogisticRegression.cpp -Iinclude/ -Llib/ -l:mlLib.a -o examples/Exe/t6