#ifndef TENSOR_H
#define TENSOR_H

#include <memory>
#include <vector>
#include <functional>
#include <set>
#include <Eigen/Dense>

/*
 *this is core engine, Tensor class
 */

class Tensor : public std::enable_shared_from_this<Tensor> {
private:
    Eigen::MatrixXf data_;
    Eigen::MatrixXf grad_;
    bool requires_grad_;
    std::string op_;
    std::set<std::shared_ptr<Tensor>> prev_;
    std::function<void()> backward_fn_;
    std::string label_;

public:
    explicit Tensor(const Eigen::MatrixXf& data, bool requires_grad = false, const std::string& label = "");

    std::shared_ptr<Tensor> matmul(std::shared_ptr<Tensor> other);
    std::shared_ptr<Tensor> add(std::shared_ptr<Tensor> other);
    std::shared_ptr<Tensor> relu();
    std::shared_ptr<Tensor> log_softmax();
    std::shared_ptr<Tensor> mse_loss(std::shared_ptr<Tensor> target);
    std::shared_ptr<Tensor> nll_loss(const std::vector<int>& target);

    void backward();

    std::shared_ptr<Tensor> reshape(int rows, int cols);
    int rows() const { return data_.rows(); }
    int cols() const { return data_.cols(); }

    Eigen::MatrixXf& data() { return data_; }
    const Eigen::MatrixXf& data() const { return data_; }
    Eigen::MatrixXf& grad() { return grad_; }
    const Eigen::MatrixXf& grad() const { return grad_; }
    void zero_grad() { if (requires_grad_) grad_.setZero(); }
    bool requires_grad() const { return requires_grad_; }
    const std::set<std::shared_ptr<Tensor>>& prev() const { return prev_; }
    const std::string& op() const { return op_; }
    void set_label(const std::string& label) { label_ = label; }
    const std::string& label() const { return label_; }
};

std::shared_ptr<Tensor> operator+(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b);
std::shared_ptr<Tensor> matmul(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b);
std::shared_ptr<Tensor> relu(std::shared_ptr<Tensor> x);

#endif // TENSOR_H