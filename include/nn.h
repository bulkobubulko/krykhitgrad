#ifndef NN_H
#define NN_H

#include "tensor.h"
#include <vector>
#include <memory>

/*
 *this is nn library built on top of engine
 */

class Module {
public:
    virtual std::vector<std::shared_ptr<Tensor>> parameters() = 0;

    void zero_grad() {
        for (auto& p : parameters()) {
            p->zero_grad();
        }
    }

    virtual ~Module() = default;
};

class Linear : public Module {
private:
    std::shared_ptr<Tensor> weight_;
    std::shared_ptr<Tensor> bias_;
    int in_features_;
    int out_features_;

public:
    Linear(int in_features, int out_features);

    std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor> x);

    std::vector<std::shared_ptr<Tensor>> parameters() override {
        return {weight_, bias_};
    }
};

class Sequential : public Module {
private:
    std::vector<std::shared_ptr<Module>> modules_;

public:
    Sequential(const std::vector<std::shared_ptr<Module>>& modules)
        : modules_(modules) {
    }

    std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor> x);

    std::vector<std::shared_ptr<Tensor>> parameters() override;
};

class ReLU : public Module {
public:
    std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor> x) {
        return x->relu();
    }

    std::vector<std::shared_ptr<Tensor>> parameters() override {
        return {};
    }
};

#endif // NN_H
