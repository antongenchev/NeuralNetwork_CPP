#include "loss.h"
#include <vector>
#include <cmath>

// MSE = (1/n) * SUM(actual[i] - predicted[i])^2
double mse(const std::vector<double>& predicted, const std::vector<double>& actual) {
    double sum_error = 0.0;
    int n = predicted.size();
    
    for (int i = 0; i < n; ++i) {
        sum_error += std::pow(predicted[i] - actual[i], 2);
    }
    
    return sum_error / n;
}

// dMSE/d(predicted[i]) = -2 * (actual[i] - predicted[i]) / n
std::vector<double> mse_derivative(const std::vector<double>& predicted, const std::vector<double>& actual) {
    int n = predicted.size();
    std::vector<double> gradient(n);
    
    for (int i = 0; i < n; ++i) {
        gradient[i] = -2.0 * (actual[i] - predicted[i]) / n;
    }
    
    return gradient;
}