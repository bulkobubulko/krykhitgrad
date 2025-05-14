#ifndef DATA_H
#define DATA_H

#include "tensor.h"
#include <string>
#include <vector>
#include <utility>

class MNISTDataset {
private:
    Eigen::MatrixXf images_;
    std::vector<int> labels_;

public:
    MNISTDataset(const std::string& images_file, const std::string& labels_file, int max_samples = -1);

    int size() const { return labels_.size(); }

    std::pair<std::shared_ptr<Tensor>, std::vector<int>> get_batch(
        int batch_size, int offset);
};

#endif // DATA_H