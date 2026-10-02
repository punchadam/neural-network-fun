#pragma once

#include "linearAlgebra.h"

#include <fstream>
#include <vector>
#include <span>
#include <cstdint>
#include <string>
#include <stdexcept>
#include <format>
#include <bit>

using u8 = uint8_t;
using u16 = uint16_t;
using u32 = uint32_t;
using u64 = uint64_t;

struct MNIST_Image {
    std::span<const u8> pixels;
    u8 label = 0;

    // extracts and normalizes pixel data into a Vector
    template <typename T, template <typename> class Vector>
    Vector<T> toVector() const {
        Vector<T> vec(pixels.size());
        for (size_t i = 0; i < pixels.size(); ++i) {
            vec[i] = static_cast<T>(pixels[i]) / static_cast<T>(255.0);
        }
        return vec;
    }

    // generates one-hot vector from label
    template <typename T, template <typename> class Vector>
    Vector<T> toTargetVector(size_t numClasses = 10) const {
        Vector<T> target(numClasses);
        // ensure all elements start at 0
        for (size_t i = 0; i < numClasses; ++i) {
            target[i] = static_cast<T>(0);
        }
        if (label < numClasses) {
            target[label] = static_cast<T>(1);
        }
        return target;
    }
};

template <typename T>
struct MNIST_Dataset {
    u32 imageHeight = 0;
    u32 imageWidth = 0;
    
    std::vector<u8> rawBytes;
    std::vector<MNIST_Image> images;
};

inline std::vector<u8> readLabels(const std::string& labelDataFileName) {
    std::ifstream labelFile(labelDataFileName, std::ios::binary);
    if (!labelFile.is_open()) {
        throw std::runtime_error(std::format("Could not open label file {}", labelDataFileName));
    }

    // Read magic number and number of items
    u32 header[2];
    for (u32& number : header) {
        labelFile.read(reinterpret_cast<char*>(&number), sizeof(number));
        number = std::byteswap(number); // convert big-endian to host endianness
    }

    // Magic number for MNIST label files is 2049
    if (header[0] != 2049) {
        throw std::invalid_argument(std::format("Incorrect or missing magic number in {}", labelDataFileName));
    }

    u32 numLabels = header[1];
    std::vector<u8> labels(numLabels);

    // Read label bytes
    labelFile.read(reinterpret_cast<char*>(labels.data()), numLabels);
    if (!labelFile) {
        throw std::runtime_error(std::format("Failed to read labels from file {}", labelDataFileName));
    }

    return labels;
}

template <typename T>
MNIST_Dataset<T> readImageData(const std::string& imageDataFileName, const std::string& labelDataFileName = "") {
    std::ifstream imageDataFile(imageDataFileName, std::ios::binary);   // open the file in binary mode
    if (!imageDataFile.is_open()) {
        throw std::runtime_error(std::format("Could not open image file {}", imageDataFileName));
    }

    // read the first 4 fields at the beginning of the file
    // 1. magic number
    // 2. number of images
    // 3. image height
    // 4. image width
    u32 header[4];
    for (u32& number : header) {
        imageDataFile.read(reinterpret_cast<char*>(&number), sizeof(number));
        number = std::byteswap(number); // big endian to little endian
    }

    if (header[0] != 2051) {
        throw std::invalid_argument(std::format("Incorrect or missing magic number in {}", imageDataFileName));
    }

    u32 numImages = header[1];
    u32 rows = header[2];
    u32 cols = header[3];
    size_t imageSize = static_cast<size_t>(rows) * cols;
    size_t totalBytes = static_cast<size_t>(numImages) * imageSize;

    MNIST_Dataset<T> dataset;
    dataset.imageHeight = rows;
    dataset.imageWidth = cols;

    // copy into dataset
    dataset.rawBytes.resize(totalBytes);
    imageDataFile.read(reinterpret_cast<char*>(dataset.rawBytes.data()), totalBytes);

    if (!imageDataFile) {
        throw std::runtime_error(std::format("Failed to read data from file {}", imageDataFileName));
    }

    std::vector<u8> rawLabels;
    if (!labelDataFileName.empty()) {
        rawLabels = readLabels(labelDataFileName);
        if (rawLabels.size() != numImages) {
            throw std::runtime_error(std::format(
                "Label count ({}) does not match image count ({})", 
                rawLabels.size(), numImages
            ));
        }
    }

    // construct std::span views into rawBytes
    dataset.images.reserve(numImages);
    const u8* dataPtr = dataset.rawBytes.data();

    for (size_t i = 0; i < numImages; ++i) {
        std::span<const u8> imageSpan(dataPtr + (i * imageSize), imageSize);
        dataset.images.push_back(MNIST_Image{
            .pixels = imageSpan,
            .label = rawLabels[i]
        });
    }

    return dataset;
}