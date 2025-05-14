#include "../include/optim.h"

void SGD::step() {
    for (auto& param : parameters_) {
        param->data() -= lr_ * param->grad();
    }
}