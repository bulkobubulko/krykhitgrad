#include "../include/nn.h"
#include <random>

Linear::Linear(int in_features, int out_features)
    : in_features_(in_features), out_features_(out_features) {

    float std_dev = std::sqrt(2.0f / (in_features + out_features)); // xavier init

    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> dist(0.0f, std_dev);

    Eigen::MatrixXf w(out_features, in_features);
    for (int i = 0; i < out_features; ++i) {
        for (int j = 0; j < in_features; ++j) {
            w(i, j) = dist(gen);
        }
    }

    weight_ = std::make_shared<Tensor>(w, true, "weight");

    Eigen::MatrixXf b = Eigen::MatrixXf::Zero(out_features, 1);
    bias_ = std::make_shared<Tensor>(b, true, "bias");
}

std::shared_ptr<Tensor> Linear::forward(std::shared_ptr<Tensor> x) {
    auto out = weight_->matmul(x);

    Eigen::MatrixXf expanded_bias = bias_->data().replicate(1, x->data().cols());
    auto bias_tensor = std::make_shared<Tensor>(expanded_bias);

    return out->add(bias_tensor);
}

std::shared_ptr<Tensor> Sequential::forward(std::shared_ptr<Tensor> x) {
    auto out = x;
    for (auto& module : modules_) {
        out = std::dynamic_pointer_cast<Linear>(module)->forward(out);
        if (module != modules_.back()) {
            out = out->relu();
        }
    }
    return out;
}

std::vector<std::shared_ptr<Tensor>> Sequential::parameters() {
    std::vector<std::shared_ptr<Tensor>> params;
    for (auto& module : modules_) {
        auto module_params = module->parameters();
        params.insert(params.end(), module_params.begin(), module_params.end());
    }
    return params;
}