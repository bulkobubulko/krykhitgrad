#ifndef VALUE_H
#define VALUE_H

#include <iostream>
#include <memory>
#include <string>
#include <set>
#include <functional>
#include <unordered_set>

class Value : public std::enable_shared_from_this<Value> {
private:
    double data_;
    double grad_ = 0.0;
    std::string op_;
    std::set<std::shared_ptr<Value>> prev_;
    std::string label_;
    std::function<void()> backward_;

public:
    explicit Value(double data, const std::string& label = ""); // leaf node

    Value(double data,
        double grad,
        const std::string& op,
        const std::set<std::shared_ptr<Value>>& prev,
        const std::string& label_);

    std::shared_ptr<Value> operator+(std::shared_ptr<Value> other);
    std::shared_ptr<Value> operator*(std::shared_ptr<Value> other);

    std::shared_ptr<Value> tanh();
    std::shared_ptr<Value> exp();
    std::shared_ptr<Value> pow(std::shared_ptr<Value> other);

    std::string str() const;

    friend std::ostream& operator<<(std::ostream& os, const Value& val);

    double get_data() const { return data_; }
    void set_grad(double grad) { grad_ = grad; }
    double get_grad() const { return grad_; }
    const std::string& get_op() const { return op_; }
    const std::set<std::shared_ptr<Value>>& get_prev() const { return prev_; }
    void set_label(std::string label) { label_ = label; }
    const std::string& get_label() const { return label_; }
    void backward() const { if (backward_) backward_(); }

    void backward();
};

#endif //VALUE_H