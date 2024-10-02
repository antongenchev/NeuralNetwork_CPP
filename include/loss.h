#pragma once

#include <vector>

double mse(const std::vector<double>& predicted, const std::vector<double>& actual);
std::vector<double> mse_derivative(const std::vector<double>& predicted, const std::vector<double>& actual);
