#include "../include/value.h"

#include <cmath>

Value::Value(double data, const std::string& label)
    : data_(data), grad_(0.0), op_(""), prev_(), label_(label) {}

Value::Value(double data, double grad, const std::string& op,
             const std::set<std::shared_ptr<Value>>& prev,
             const std::string& label)
    : data_(data), grad_(grad), op_(op), prev_(prev), label_(label) {}

std::shared_ptr<Value> Value::operator+(std::shared_ptr<Value> other) {
    double result = this->data_ + other->data_;
    std::string op = "+";
    std::set<std::shared_ptr<Value>> prev = { shared_from_this(), other };
    std::string label = "(" + this->label_ + "+" + other->label_ + ")";

    auto out = std::make_shared<Value>(result, 0.0, op, prev, label);

    out->backward_ = [this, other, out]() {
        auto self = shared_from_this();
        this->grad_ += 1.0 * out->grad_;
        other->grad_ += 1.0 * out->grad_;
    };

    return out;
}

std::shared_ptr<Value> Value::operator*(std::shared_ptr<Value> other) {
    double result = this->data_ * other->data_;
    std::string op = "*";
    std::set<std::shared_ptr<Value>> prev = { shared_from_this(), other };
    std::string label = "(" + this->label_ + op + other->label_ + ")";

    auto out = std::make_shared<Value>(result, 0.0, op, prev, label);

    out->backward_ = [this, other, out]() {
        auto self = shared_from_this();
        this->grad_ += other->data_ * out->grad_;
        other->grad_ += this->data_ * out->grad_;
    };

    return out;
}

std::shared_ptr<Value> Value::tanh() {
    double result = (std::exp(2*this->data_) - 1) / (std::exp(2*this->data_) + 1);
    std::string op = "tanh";
    std::set<std::shared_ptr<Value>> prev = { shared_from_this() };
    std::string label = "(" + op + "(" + this->label_ + ")" + ")";

    auto out = std::make_shared<Value>(result, 0.0, op, prev, label);

    double saved_result = result;
    out->backward_ = [this, out, saved_result]() {
        auto self = shared_from_this();
        this->grad_ += (1 - std::pow(saved_result, 2)) * out->grad_;
    };

    return out;
}

std::shared_ptr<Value> Value::exp() {
    double result = std::exp(this->data_);
    std::string op = "exp";
    std::set<std::shared_ptr<Value>> prev = { shared_from_this() };
    std::string label = "(" + op + "(" + this->label_ + ")" + ")";

    auto out = std::make_shared<Value>(result, 0.0, op, prev, label);

    out->backward_ = [this, out]() {
        auto self = shared_from_this();
        this->grad_ += out->data_ * out->grad_;
    };

    return out;
}

std::shared_ptr<Value> Value::pow(std::shared_ptr<Value> other) {
    double result = std::pow(this->data_, other->data_);
    std::string op = "pow";
    std::set<std::shared_ptr<Value>> prev = { shared_from_this() };
    std::string label = "(" + op + "(" + this->label_ + "," + other->label_ + ")" + ")";

    auto out = std::make_shared<Value>(result, 0.0, op, prev, label);

    out->backward_ = [this, out, other]() {
        auto self = shared_from_this();
        this->grad_ += other->data_ * std::pow(this->data_, other->data_ - 1) * out->grad_;
    };

    return out;
}

std::string Value::str() const {
    return "Value(data=" + std::to_string(data_) + ", label=\"" + label_ + "\")";
}

std::ostream& operator<<(std::ostream& os, const Value& val) {
    os << val.str();
    return os;
}

void Value::backward() {
    std::vector<std::shared_ptr<Value>> topo;
    std::unordered_set<Value*> visited;

    std::function<void(std::shared_ptr<Value>)> build_topo = [&](std::shared_ptr<Value> v) {
        if (visited.find(v.get()) == visited.end()) {
            visited.insert(v.get());
            for (const auto& child : v->prev_) {
                build_topo(child);
            }
            topo.push_back(v);
        }
    };

    build_topo(shared_from_this());
    this->grad_ = 1.0;

    std::reverse(topo.begin(), topo.end());
    for (const auto& v : topo) {
        if (v->backward_) v->backward_();
    }
}