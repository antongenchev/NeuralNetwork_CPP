#include "utils.h"
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <iostream>
#include <algorithm>
#include <random>

std::vector<std::vector<double>> readCSV(const std::string& filename) {
    std::vector<std::vector<double>> data;
    std::ifstream file(filename);

    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filename << std::endl;
        return data;
    }
    std::string line;
    while (std::getline(file, line)) {
        std::vector<double> row;
        std::stringstream ss(line);
        std::string value;
        while (std::getline(ss, value, ',')) {
            try {
                row.push_back(std::stod(value));
            } catch (const std::invalid_argument& e) {
                std::cerr << "Error: Invalid number found in file " << filename << " - " << value << std::endl;
                continue;
            }
        }
        if (!row.empty()) {
            data.push_back(row);
        }
    }
    file.close();
    return data;
}

void shuffle_data(std::vector<std::vector<double>>& X, std::vector<std::vector<double>>& y) {
    if (X.size() != y.size()) {
        std::cerr << "Error: Size mismatch between X and y. Cannot shuffle data." << std::endl;
        return;
    }

    // Create a vector of indices
    std::vector<size_t> indices(X.size());
    for (size_t i = 0; i < indices.size(); ++i) {
        indices[i] = i;
    }
    // Shuffle the indices
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(indices.begin(), indices.end(), g);

    // Use the shuffled indices to rearrange X and y
    std::vector<std::vector<double>> X_shuffled(X.size());
    std::vector<std::vector<double>> y_shuffled(y.size());
    for (size_t i = 0; i < indices.size(); ++i) {
        X_shuffled[i] = X[indices[i]];
        y_shuffled[i] = y[indices[i]];
    }

    X = X_shuffled;
    y = y_shuffled;
}