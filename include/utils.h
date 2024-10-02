#pragma once

#include <vector>
#include <string>

std::vector<std::vector<double>> readCSV(const std::string& filename);
void shuffle_data(std::vector<std::vector<double>>& X, std::vector<std::vector<double>>& y);
