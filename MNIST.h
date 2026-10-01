#pragma once

#include <fstream>
#include <cstdint>
#include <stdexcept>
#include <iostream>
#include <string>

void readFile(const std::string& fileName) {
    std::string text;
    std::ifstream myFile(fileName);
    while (std::getline(myFile, text)) {
        std::cout << text;
    }
    myFile.close();
}