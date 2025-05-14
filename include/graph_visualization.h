#ifndef GRAPH_VISUALIZATION_H
#define GRAPH_VISUALIZATION_H

#include <string>
#include <fstream>
#include <sstream>
#include <memory>
#include <set>
#include <unordered_set>
#include <unordered_map>
#include "value.h"

class GraphVisualizer {
private:
    void trace(std::shared_ptr<Value> root,
               std::unordered_set<std::shared_ptr<Value>>& nodes,
               std::set<std::pair<std::shared_ptr<Value>, std::shared_ptr<Value>>>& edges) {

        if (nodes.find(root) == nodes.end()) {
            nodes.insert(root);
            for (const auto& child : root->get_prev()) {
                edges.insert({child, root});
                trace(child, nodes, edges);
            }
        }
    }

public:
    std::string draw_dot(std::shared_ptr<Value> root, const std::string& format = "svg", const std::string& rankdir = "LR") {
        std::unordered_set<std::shared_ptr<Value>> nodes;
        std::set<std::pair<std::shared_ptr<Value>, std::shared_ptr<Value>>> edges;

        trace(root, nodes, edges);

        std::stringstream dot;
        dot << "digraph {" << std::endl;
        dot << "  rankdir=" << rankdir << ";" << std::endl;

        for (const auto& n : nodes) {
            dot << "  \"" << n.get() << "\" [shape=record, label=\"{ "
                << n->get_label() << " | data " << n->get_data()
                << " | grad " << n->get_grad() << " }\"];" << std::endl;

            if (!n->get_op().empty()) {
                dot << "  \"" << n.get() << n->get_op() << "\" [label=\"" << n->get_op() << "\"];" << std::endl;
                dot << "  \"" << n.get() << n->get_op() << "\" -> \"" << n.get() << "\";" << std::endl;
            }
        }

        for (const auto& [n1, n2] : edges) {
            dot << "  \"" << n1.get() << "\" -> \"" << n2.get() << n2->get_op() << "\";" << std::endl;
        }

        dot << "}" << std::endl;
        return dot.str();
    }

    bool save_dot(std::shared_ptr<Value> root, const std::string& filename,
                  const std::string& format = "svg", const std::string& rankdir = "LR") {
        std::string dot_content = draw_dot(root, format, rankdir);

        std::ofstream file(filename + ".dot"); // .dot, .svg
        if (!file.is_open()) {
            return false;
        }

        file << dot_content;
        file.close();

        std::string cmd = "dot -T" + format + " " + filename + ".dot -o " + filename + "." + format;
        int result = system(cmd.c_str());

        return result == 0;
    }
};

#endif // GRAPH_VISUALIZATION_H