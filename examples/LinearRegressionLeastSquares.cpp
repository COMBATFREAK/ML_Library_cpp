#include "../include/mlLib.h"
#include <iostream>
#include <vector>

int main()
{
    const int vectorSize = 100000;

    // Create vectors for x and y with the specified size
    std::vector<double> xValues(vectorSize);
    std::vector<double> yValues(vectorSize);

    // Fill xValues and yValues with some example data
    // For simplicity, I'm using a linear relationship with some noise
    for (int i = 0; i < vectorSize; ++i)
    {
        xValues[i] = i;
        yValues[i] = 2.5 * i + 100.0 + (rand() % 50 - 5); // Linear relationship with noise
    }

    // Split the data into training and test sets (80% training, 20% test)
    const int trainingSize = static_cast<int>(vectorSize * 0.8);

    std::vector<double> xTrain(xValues.begin(), xValues.begin() + trainingSize);
    std::vector<double> yTrain(yValues.begin(), yValues.begin() + trainingSize);

    std::vector<double> xTest(xValues.begin() + trainingSize, xValues.end());
    std::vector<double> yTest(yValues.begin() + trainingSize, yValues.end());

    // Train the linear regression model
    mlLib::LinearRegressionModel model = mlLib::LinearRegressionLeastSquares(xTrain, yTrain,stat::Min_Max);

    // Predict using the test set
    std::vector<long double> predictedYValues = model.predict(xTest);

    // Evaluate the model using Mean Squared Error (MSE)
    long double mse = model.evaluate(yTest, predictedYValues);

    // Print the MSE
    std::cout << "Mean Squared Error (MSE) on Test Set: " << mse << std::endl;

    // Print model parameters
    std::cout << "Linear Regression Model Parameters:" << std::endl;
    std::cout << "Slope: " << model.getSlope() << std::endl;
    std::cout << "Intercept: " << model.getIntercept() << std::endl;

    return 0;
}