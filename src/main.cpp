// This is a personal academic project. Dear PVS-Studio, please check it.
// PVS-Studio Static Code Analyzer for C, C++, C#, and Java: http://www.viva64.com

#include <iostream>
#include <Eigen/Eigen>
#include "options_parser.h"
#include "../include/value.h"
#include "../include/tensor.h"
#include "../include/nn.h"
#include "../include/graph_visualization.h"


int main() {
    auto x1 = std::make_shared<Value>(0.5, "x1");
    auto x2 = std::make_shared<Value>(-2.0, "x2");

    auto w1 = std::make_shared<Value>(0.2, "w1");
    auto w2 = std::make_shared<Value>(-0.5, "w2");
    auto b1 = std::make_shared<Value>(0.1, "b1");

    auto z1 = *w1 * x1;
    z1->set_label("z1");
    auto z2 = *w2 * x2;
    z2->set_label("z2");

    auto z = *z1 + z2;
    z->set_label("z");

    auto h = (*z + b1)->tanh();
    h->set_label("h");

    h->backward();

    GraphVisualizer visualizer;
    visualizer.save_dot(h, "h");
    return 0;
}
