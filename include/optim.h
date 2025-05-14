#ifndef OPTIM_H
#define OPTIM_H

#include "tensor.h"
#include <vector>
#include <memory>

class Optimizer {
protected:
    std::vector<std::shared_ptr<Tensor>> parameters_;

public:
    explicit Optimizer(const std::vector<std::shared_ptr<Tensor>>& parameters)
        : parameters_(parameters) {
    }

    virtual void step() = 0;

    void zero_grad() {
        for (auto& p : parameters_) {
            p->zero_grad();
        }
    }

    virtual ~Optimizer() = default;
};

class SGD : public Optimizer {
private:
    float lr_;

public:
    SGD(const std::vector<std::shared_ptr<Tensor>>& parameters, float learning_rate = 0.01)
        : Optimizer(parameters), lr_(learning_rate) {
    }

    void step() override;
};

#endif // OPTIM_H
