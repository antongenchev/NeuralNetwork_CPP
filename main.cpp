#include "neural_network.h"
#include "utils.h"
#include <iostream>

int main() {
    // Load the data
    std::string filename = "data/data.csv";
    std::vector<std::vector<double>> data = readCSV(filename);

    std::vector<std::vector<double>> X_train;
    std::vector<std::vector<double>> y_train;

    for (const auto& row : data) {
        if (row.size() >= 8) {
            y_train.push_back({row[1]});
            X_train.push_back({row[0], row[2], row[3], row[4], row[5], row[6], row[7]});
        }
    }

    // Create a model with one hidden layer of 16 neurons
    NeuralNetwork nn;
    nn.addLayer(7, 16);
    nn.addLayer(16, 16);
    nn.addLayer(16, 1);
    // Train the model
    int epochs = 1000;
    double learning_rate = 0.001;
    nn.train(X_train, y_train, epochs, learning_rate);

    std::vector<std::vector<double>> test_inputs = {
        {18.0, 307.0, 130.0, 3504.0, 12.0, 70, 1},
        {15.0, 350.0, 165.0, 3693.0, 11.5, 70, 1}
    };

    for (const auto& input : test_inputs) {
        std::vector<double> prediction = nn.predict(input);
        std::cout << "Input: {" << input[0] << ", " << input[1] << ", " << input[2] << ", " << input[3] << ", " << input[4] << ", " << input[5] << ", " << input[6] << "} -> Prediction: " << prediction[0] << std::endl;
    }
    return 0;
}