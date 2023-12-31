#include "../../include/mlLib.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>

// Function to split a string into tokens based on a delimiter
std::vector<std::string> split(const std::string &s, char delimiter)
{
    std::vector<std::string> tokens;
    std::istringstream ss(s);
    std::string token;
    while (std::getline(ss, token, delimiter))
    {
        tokens.push_back(token);
    }
    return tokens;
}

int main()
{
    // Open the CSV file
    std::ifstream file("examples/DataSet/TaxiFare.csv");

    if (!file.is_open())
    {
        std::cerr << "Error opening the file!" << std::endl;
        return 1;
    }

    // Create vectors to store data
    std::vector<double> xValues;
    std::vector<double> yValues;

    // Read and parse the CSV file
    std::string line;
    std::getline(file, line); // Skip header line

    while (std::getline(file, line))
    {
        // Split the line into tokens
        std::vector<std::string> tokens = split(line, ',');

        // Ensure we have at least 8 tokens (adjust if needed)
        if (tokens.size() >= 8)
        {
            // Extract trip_duration and total_fare
            double tripDuration = std::stod(tokens[0]);
            double totalFare = std::stod(tokens[6]);

            // Store in vectors
            xValues.push_back(tripDuration);
            yValues.push_back(totalFare);
        }
        else
        {
            std::cerr << "Invalid number of columns in the CSV file." << std::endl;
        }
    }

    const size_t vectorSize = xValues.size();

    const int trainingSize = static_cast<int>(vectorSize * 0.8);

    std::vector<double> xTrain(xValues.begin(), xValues.begin() + trainingSize);
    std::vector<double> yTrain(yValues.begin(), yValues.begin() + trainingSize);

    std::vector<double> xTest(xValues.begin() + trainingSize, xValues.end());
    std::vector<double> yTest(yValues.begin() + trainingSize, yValues.end());

    // Train the linear regression model using gradient descent
    mlLib::LinearRegressionModel model = mlLib::LinearRegressionGradientDescent(xTrain, yTrain, stat::NormalizationType::L2_Euclidean, 0.05, 1000);

    // Predict using the test set
    const std::vector<long double> predictedYValues = model.predict(xTest);

    // Evaluate the model using Mean Squared Error (MSE)
    const long double mse = model.evaluate(yTest, predictedYValues);

    // Print the MSE
    std::cout << "Linear Regression Gradient Descent\n";
    std::cout << "Mean Squared Error (MSE) on Test Set: " << mse << std::endl;

    // Print model parameters
    std::cout << "Linear Regression Model Parameters:" << std::endl;
    std::cout << "Slope: " << model.getSlope() << std::endl;
    std::cout << "Intercept: " << model.getIntercept() << std::endl;

    // Close the file
    file.close();

    return 0;
}
