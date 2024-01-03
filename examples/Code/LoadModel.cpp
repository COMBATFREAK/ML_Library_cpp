#include "../../include/mlLib.h"
#include <iostream>

int main()
{

    mlLib::LinearRegressionModel model1 = mlLib::LinearRegressionModel();
    mlLib::LinearRegressionModel model2 = mlLib::LinearRegressionModel();
    mlLib::LogisticRegressionModel model3 = mlLib::LogisticRegressionModel();

    model1.loadFromFile("examples/Models/LinRegLeastSqModel.txt");
    model2.loadFromFile("examples/Models/LinRegGradDesModel.txt");
    model3.loadFromFile("examples/Models/LogRegModel.txt");

    model1.printInfo();
    std::cout << "\n-------------------------------------------------\n";
    model2.printInfo();
    std::cout << "\n-------------------------------------------------\n";
    model3.printInfo();
}