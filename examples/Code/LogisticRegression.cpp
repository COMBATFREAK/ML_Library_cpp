#include "../../include/mlLib.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <iomanip>

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
    std::ifstream file("examples/DataSet/HeartDisease.csv");

    if (!file.is_open())
    {
        std::cerr << "Error opening the file!" << std::endl;
        return 1;
    }

    // Create vectors to store data
    std::vector<std::vector<double>> xValues;
    std::vector<double> yValues;

    // Read and parse the CSV file
    std::string line;
    std::getline(file, line); // Skip header line

    while (std::getline(file, line))
    {
        // Split the line into tokens
        std::vector<std::string> tokens = split(line, ',');

        // Ensure we have at least 16 tokens (adjust if needed)
        if (tokens.size() >= 16)
        {
            // Extract values
            std::vector<double> rowValues;
            for (int i = 0; i < 15; ++i)
            {
                rowValues.push_back(std::stod(tokens[i]));
            }

            // Store in vectors
            xValues.push_back(rowValues);
            yValues.push_back(std::stod(tokens[15]));
        }
        else
        {
            std::cerr << "Invalid number of columns in the CSV file." << std::endl;
            return 1;
        }
    }

    const size_t vectorSize = yValues.size();
    const int trainingSize = static_cast<int>(vectorSize * 0.5);

    std::vector<std::vector<double>> xTrain(xValues.begin(), xValues.begin() + trainingSize);
    std::vector<std::vector<double>> xTest(xValues.begin() + trainingSize, xValues.end());

    std::vector<double> yTrain(yValues.begin(), yValues.begin() + trainingSize);
    std::vector<double> yTest(yValues.begin() + trainingSize, yValues.end());

    mlLib::LogisticRegressionModel model = mlLib::LogisticRegression(xTrain, yTrain,0.0005,100000);

    std::vector<int> predictedYValues = model.predict(xTest,0.9);

    const long double Accuracy = model.evaluate(yTest, predictedYValues);

    std::cout << "Accuracy = " << Accuracy << "%\n";

    mlLib::ConfusionMatrix confusionMatrix = model.getConfusionMatrix();
    mlLib::EvaluationMetrics evaluationMetrics = model.getEvaluationMetrics();

    // Print confusion matrix
    std::cout << "Confusion Matrix:\n";
    std::cout << std::setw(15) << "True Positive: " << confusionMatrix.truePositive << "\n";
    std::cout << std::setw(15) << "True Negative: " << confusionMatrix.trueNegative << "\n";
    std::cout << std::setw(15) << "False Positive: " << confusionMatrix.falsePositive << "\n";
    std::cout << std::setw(15) << "False Negative: " << confusionMatrix.falseNegative << "\n\n";

    // Print evaluation metrics
    std::cout << "Evaluation Metrics:\n";
    std::cout << std::setw(15) << "Accuracy: " << evaluationMetrics.accuracy << "%\n";
    std::cout << std::setw(15) << "Recall: " << evaluationMetrics.recall << "%\n";
    std::cout << std::setw(15) << "Precision: " << evaluationMetrics.precision << "%\n";
    std::cout << std::setw(15) << "F1 Score: " << evaluationMetrics.f1Score << "\n";

    // Close the file
    file.close();
    return 0;
}
