#include "../include/tensor.h"
#include <algorithm>
#include <iostream>
#include <unordered_set>

Tensor::Tensor(const Eigen::MatrixXf& data, bool requires_grad, const std::string& label)
    : data_(data), requires_grad_(requires_grad), label_(label) {
    if (requires_grad) {
        grad_ = Eigen::MatrixXf::Zero(data.rows(), data.cols());
    }
}

std::shared_ptr<Tensor> Tensor::matmul(std::shared_ptr<Tensor> other) {
    Eigen::MatrixXf result = data_ * other->data_;
    auto out = std::make_shared<Tensor>(result, requires_grad_ || other->requires_grad_);

    if (requires_grad_ || other->requires_grad_) {
        out->prev_ = {shared_from_this(), other};
        out->op_ = "matmul";

        out->backward_fn_ = [self=shared_from_this(), other, out]() {
            if (self->requires_grad_) {
                self->grad_ += out->grad_ * other->data_.transpose();
            }
            if (other->requires_grad_) {
                other->grad_ += self->data_.transpose() * out->grad_;
            }
        };
    }

    return out;
}

std::shared_ptr<Tensor> Tensor::add(std::shared_ptr<Tensor> other) {
    Eigen::MatrixXf result = data_ + other->data_;
    auto out = std::make_shared<Tensor>(result, requires_grad_ || other->requires_grad_);

    if (requires_grad_ || other->requires_grad_) {
        out->prev_ = {shared_from_this(), other};
        out->op_ = "add";

        out->backward_fn_ = [self=shared_from_this(), other, out]() {
            if (self->requires_grad_) {
                if (self->data_.rows() == out->grad_.rows() && self->data_.cols() == out->grad_.cols()) {
                    self->grad_ += out->grad_;
                } else {
                    Eigen::MatrixXf sum = Eigen::MatrixXf::Zero(self->data_.rows(), self->data_.cols());
                    for (int i = 0; i < out->grad_.cols(); i++) {
                        sum += out->grad_.col(i);
                    }
                    self->grad_ += sum;
                }
            }

            if (other->requires_grad_) {
                if (other->data_.rows() == out->grad_.rows() && other->data_.cols() == out->grad_.cols()) {
                    other->grad_ += out->grad_;
                } else {
                    Eigen::MatrixXf sum = Eigen::MatrixXf::Zero(other->data_.rows(), other->data_.cols());
                    for (int i = 0; i < out->grad_.cols(); i++) {
                        sum += out->grad_.col(i);
                    }
                    other->grad_ += sum;
                }
            }
        };
    }

    return out;
}

std::shared_ptr<Tensor> Tensor::relu() {
    Eigen::MatrixXf result = data_.array().max(0.0f);
    auto out = std::make_shared<Tensor>(result, requires_grad_);

    if (requires_grad_) {
        out->prev_ = {shared_from_this()};
        out->op_ = "relu";

        out->backward_fn_ = [self=shared_from_this(), out]() {
            Eigen::MatrixXf mask = (self->data_.array() > 0.0f).cast<float>();
            self->grad_ += (mask.array() * out->grad_.array()).matrix();
        };
    }

    return out;
}

std::shared_ptr<Tensor> Tensor::log_softmax() {
    Eigen::MatrixXf logits = data_;
    Eigen::VectorXf max_logits = logits.colwise().maxCoeff();

    for (int i = 0; i < logits.cols(); i++) {
        logits.col(i).array() -= max_logits(i);
    }

    Eigen::MatrixXf exp_logits = logits.array().exp();
    Eigen::VectorXf sum_exp = exp_logits.colwise().sum();

    Eigen::MatrixXf log_softmax_out = logits;
    for (int i = 0; i < logits.cols(); i++) {
        log_softmax_out.col(i).array() -= std::log(sum_exp(i));
    }

    auto out = std::make_shared<Tensor>(log_softmax_out, requires_grad_);

    if (requires_grad_) {
        out->prev_ = {shared_from_this()};
        out->op_ = "log_softmax";

        out->backward_fn_ = [self=shared_from_this(), out, exp_logits, sum_exp]() {
            Eigen::MatrixXf softmax = exp_logits;
            for (int i = 0; i < softmax.cols(); i++) {
                softmax.col(i) /= sum_exp(i);
            }

            for (int i = 0; i < out->grad_.cols(); i++) {
                float sum_grad = out->grad_.col(i).sum();
                self->grad_.col(i) += out->grad_.col(i) - softmax.col(i) * sum_grad;
            }
        };
    }

    return out;
}

std::shared_ptr<Tensor> Tensor::mse_loss(std::shared_ptr<Tensor> target) {
    int batch_size = data_.cols();
    Eigen::MatrixXf diff = data_ - target->data_;
    Eigen::MatrixXf result(1, 1);
    result(0, 0) = diff.array().square().sum() / batch_size;

    auto out = std::make_shared<Tensor>(result, requires_grad_);

    if (requires_grad_) {
        out->prev_ = {shared_from_this(), target};
        out->op_ = "mse_loss";

        out->backward_fn_ = [self=shared_from_this(), target, diff, batch_size, out]() {
            self->grad_ += 2.0f * diff * (out->grad_(0, 0) / batch_size);
        };
    }

    return out;
}

std::shared_ptr<Tensor> Tensor::nll_loss(const std::vector<int>& target) {
    int batch_size = data_.cols();
    Eigen::MatrixXf result(1, 1);
    result(0, 0) = 0.0f;

    for (int i = 0; i < batch_size; i++) {
        result(0, 0) -= data_(target[i], i);
    }
    result(0, 0) /= batch_size;

    auto out = std::make_shared<Tensor>(result, requires_grad_);

    if (requires_grad_) {
        out->prev_ = {shared_from_this()};
        out->op_ = "nll_loss";

        out->backward_fn_ = [self=shared_from_this(), target, batch_size, out]() {
            Eigen::MatrixXf grad = Eigen::MatrixXf::Zero(self->data_.rows(), self->data_.cols());

            for (int i = 0; i < batch_size; i++) {
                grad(target[i], i) = -1.0f;
            }

            self->grad_ += grad * (out->grad_(0, 0) / batch_size);
        };
    }

    return out;
}

std::shared_ptr<Tensor> Tensor::reshape(int rows, int cols) {
    Eigen::MatrixXf reshaped = Eigen::Map<Eigen::MatrixXf>(
        data_.data(), rows, cols
    );

    auto out = std::make_shared<Tensor>(reshaped, requires_grad_);

    if (requires_grad_) {
        out->prev_ = {shared_from_this()};
        out->op_ = "reshape";

        out->backward_fn_ = [self=shared_from_this(), out]() {
            Eigen::MatrixXf reshaped_grad = Eigen::Map<Eigen::MatrixXf>(
                out->grad_.data(), self->data_.rows(), self->data_.cols()
            );
            self->grad_ += reshaped_grad;
        };
    }

    return out;
}

void Tensor::backward() {
    std::vector<std::shared_ptr<Tensor>> topo;
    std::unordered_set<Tensor*> visited;

    std::function<void(std::shared_ptr<Tensor>)> build_topo = [&](std::shared_ptr<Tensor> node) {
        if (visited.find(node.get()) == visited.end()) {
            visited.insert(node.get());
            for (const auto& child : node->prev_) {
                build_topo(child);
            }
            topo.push_back(node);
        }
    };

    build_topo(shared_from_this());

    if (data_.rows() == 1 && data_.cols() == 1) {
        grad_(0, 0) = 1.0f;
    } else {
        throw std::runtime_error("backward should be called only on scalar outputs, i.e., loss)");
    }

    std::reverse(topo.begin(), topo.end());
    for (const auto& node : topo) {
        if (node->backward_fn_) {
            node->backward_fn_();
        }
    }
}

std::shared_ptr<Tensor> operator+(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b) {
    return a->add(b);
}

std::shared_ptr<Tensor> matmul(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b) {
    return a->matmul(b);
}

std::shared_ptr<Tensor> relu(std::shared_ptr<Tensor> x) {
    return x->relu();
}