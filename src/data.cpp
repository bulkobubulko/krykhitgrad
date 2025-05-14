#include "../include/data.h"

#include <fstream>
#include <stdexcept>
#include <iostream>

uint32_t swap_endian(uint32_t val) {
    // swap endianness of 32-bit integers
    return ((val << 24) & 0xFF000000) |
           ((val << 8)  & 0x00FF0000) |
           ((val >> 8)  & 0x0000FF00) |
           ((val >> 24) & 0x000000FF);
}

MNISTDataset::MNISTDataset(const std::string& images_file, const std::string& labels_file, int max_samples) {
    std::ifstream img_file(images_file, std::ios::binary);
    if (!img_file) {
        throw std::runtime_error("Cannot open file: " + images_file);
    }

    uint32_t magic, num_images, rows, cols;
    img_file.read(reinterpret_cast<char*>(&magic), 4);
    img_file.read(reinterpret_cast<char*>(&num_images), 4);
    img_file.read(reinterpret_cast<char*>(&rows), 4);
    img_file.read(reinterpret_cast<char*>(&cols), 4);

    magic = swap_endian(magic);
    num_images = swap_endian(num_images);
    rows = swap_endian(rows);
    cols = swap_endian(cols);

    if (magic != 0x803) {
        throw std::runtime_error("invalid MNIST image file format");
    }

    if (max_samples > 0 && max_samples < static_cast<int>(num_images)) {
        num_images = max_samples;
    }

    int img_size = rows * cols;
    images_.resize(img_size, num_images);

    std::vector<unsigned char> buffer(img_size);
    for (uint32_t i = 0; i < num_images; i++) {
        img_file.read(reinterpret_cast<char*>(buffer.data()), img_size);
        for (int j = 0; j < img_size; j++) {
            images_(j, i) = static_cast<float>(buffer[j]) / 255.0f; // normalize
        }
    }

    std::ifstream label_file(labels_file, std::ios::binary);
    if (!label_file) {
        throw std::runtime_error("cannot open file: " + labels_file);
    }

    uint32_t label_magic, num_labels;
    label_file.read(reinterpret_cast<char*>(&label_magic), 4);
    label_file.read(reinterpret_cast<char*>(&num_labels), 4);

    label_magic = swap_endian(label_magic);
    num_labels = swap_endian(num_labels);

    if (label_magic != 0x801) {
        throw std::runtime_error("invalid MNIST label file format");
    }

    if (num_labels < num_images) {
        throw std::runtime_error("number of labels is less than number of images");
    }

    labels_.resize(num_images);
    for (uint32_t i = 0; i < num_images; i++) {
        unsigned char label;
        label_file.read(reinterpret_cast<char*>(&label), 1);
        labels_[i] = static_cast<int>(label);
    }
}

std::pair<std::shared_ptr<Tensor>, std::vector<int>> MNISTDataset::get_batch(
    int batch_size, int offset) {

    int actual_batch_size = std::min(batch_size, static_cast<int>(labels_.size() - offset));

    if (actual_batch_size <= 0) {
        throw std::runtime_error("invalid batch: offset out of range or batch_size <= 0");
    }

    Eigen::MatrixXf batch_images = images_.block(0, offset, images_.rows(), actual_batch_size);
    std::vector<int> batch_labels(labels_.begin() + offset,
                                 labels_.begin() + offset + actual_batch_size);

    return {std::make_shared<Tensor>(batch_images), batch_labels};
}