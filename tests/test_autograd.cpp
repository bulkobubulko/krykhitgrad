#include "../include/tensor.h"
#include "../include/nn.h"
#include "../include/optim.h"
#include <iostream>
#include <cassert>s

void test_basic_operations() {
    Eigen::MatrixXf a_data(2, 2);
    a_data << 1, 2, 3, 4;

    Eigen::MatrixXf b_data(2, 2);
    b_data << 5, 6, 7, 8;

    std::shared_ptr<Tensor> a = std::make_shared<Tensor>(a_data);
    std::shared_ptr<Tensor> b = std::make_shared<Tensor>(b_data);

    std::shared_ptr<Tensor> c = a->add(b);
    assert(c->data()(0, 0) == 6);
    assert(c->data()(1, 1) == 12);

    std::shared_ptr<Tensor> d = a->matmul(b);
    assert(d->data()(0, 0) == 19);
    assert(d->data()(1, 1) == 50);

    std::cout << "test_basic_operations: PASSED" << std::endl;
}

void test_simple_network() {
    Linear layer(2, 1);

    Eigen::MatrixXf input_data(2, 1);
    input_data << 1, 2;
    std::shared_ptr<Tensor> input = std::make_shared<Tensor>(input_data);
    std::shared_ptr<Tensor> output = layer.forward(input);

    Eigen::MatrixXf target_data(1, 1);
    target_data << 1;
    std::shared_ptr<Tensor> target = std::make_shared<Tensor>(target_data);

    std::shared_ptr<Tensor> loss = output->mse_loss(target);
    auto params = layer.parameters();
    float initial_grad_norm = params[0]->grad().norm();
    loss->backward();
    float final_grad_norm = params[0]->grad().norm();
    assert(final_grad_norm > initial_grad_norm);

    std::cout << "test_simple_network: PASSED" << std::endl;
}

void test_optimization() {
    Eigen::MatrixXf data(2, 1);
    data << 1, 2;

    std::shared_ptr<Tensor> t = std::make_shared<Tensor>(data, true);

    Eigen::MatrixXf grad(2, 1);
    grad << 0.1, 0.2;
    t->grad() = grad;

    SGD optimizer({t}, 0.1f);
    float original_value = t->data()(0, 0);
    optimizer.step();
    assert(t->data()(0, 0) != original_value);

    std::cout << "test_optimization: PASSED" << std::endl;
}

int main() {
    std::cout << "running tests for krykhitgrad..." << std::endl;

    test_basic_operations();
    test_simple_network();
    test_optimization();

    std::cout << "all tests passed!" << std::endl;
    return 0;
}
