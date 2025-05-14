#include "../include/tensor.h"
#include "../include/nn.h"
#include "../include/optim.h"
#include "../include/data.h"
#include <indicators/progress_bar.hpp>
#include <indicators/cursor_control.hpp>
#include <iostream>
#include <fstream>
#include <chrono>
#include <algorithm>
#include <vector>

class MNISTNet : public Module {
private:
    Linear fc1_;
    Linear fc2_;

public:
    MNISTNet() : fc1_(784, 128), fc2_(128, 10) {
    }

    std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor> x) {
        x = fc1_.forward(x);
        x = x->relu();
        x = fc2_.forward(x);
        return x;
    }

    std::vector<std::shared_ptr<Tensor>> parameters() override {
        auto p1 = fc1_.parameters();
        auto p2 = fc2_.parameters();
        p1.insert(p1.end(), p2.begin(), p2.end());
        return p1;
    }
};

void save_image(const std::string& filename, const Eigen::MatrixXf& image) {
    std::ofstream file(filename);
    file << "P2\n28 28\n255\n";
    for (int i = 0; i < 28; i++) {
        for (int j = 0; j < 28; j++) {
            int pixel = static_cast<int>(image(i * 28 + j) * 255.0f);
            file << pixel << " ";
        }
        file << "\n";
    }
}

void save_metrics(const std::vector<float>& train_loss,
                  const std::vector<float>& train_acc,
                  const std::vector<float>& test_acc) {
    std::ofstream file("metrics.csv");
    file << "epoch,train_loss,train_acc,test_acc\n";
    for (size_t i = 0; i < train_loss.size(); i++) {
        file << i + 1 << "," << train_loss[i] << ","
            << train_acc[i] << "," << test_acc[i] << "\n";
    }
}

float test(MNISTNet& model, const std::string& images_file, const std::string& labels_file) {
    MNISTDataset test_data(images_file, labels_file, 1000);
    int batch_size = 100;
    int num_batches = (test_data.size() + batch_size - 1) / batch_size;
    int correct = 0;
    int total = 0;
    std::vector<Eigen::MatrixXf> images;
    std::vector<int> true_labels;
    std::vector<int> pred_labels;

    indicators::show_console_cursor(false);
    indicators::ProgressBar bar{
        indicators::option::BarWidth{50},
        indicators::option::Start{"["},
        indicators::option::Fill{"="},
        indicators::option::Lead{">"},
        indicators::option::Remainder{" "},
        indicators::option::End{"]"},
        indicators::option::ForegroundColor{indicators::Color::cyan},
        indicators::option::ShowPercentage{true},
        indicators::option::MaxProgress{num_batches}
    };

    for (int batch = 0; batch < num_batches; batch++) {
        auto [inputs, targets] = test_data.get_batch(batch_size, batch * batch_size);
        auto outputs = model.forward(inputs);

        const Eigen::MatrixXf& probs = outputs->data();
        for (int i = 0; i < targets.size(); i++) {
            Eigen::MatrixXf::Index predicted;
            probs.col(i).maxCoeff(&predicted);

            if (images.size() < 10 && batch == 0) {
                images.push_back(inputs->data().col(i));
                true_labels.push_back(targets[i]);
                pred_labels.push_back(predicted);
            }

            if (predicted == targets[i]) correct++;
            total++;
        }

        bar.tick();
    }

    indicators::show_console_cursor(true);

    for (size_t i = 0; i < images.size(); i++) {
        std::string filename = "digit_true_" + std::to_string(true_labels[i]) +
            "_pred_" + std::to_string(pred_labels[i]) + ".pgm";
        save_image(filename, images[i]);
    }

    return 100.0f * correct / total;
}

int main() {
    MNISTDataset train_data("../data/mnist/train-images.idx3-ubyte", "../data/mnist/train-labels.idx1-ubyte", 10000);

    MNISTNet model;
    SGD optimizer(model.parameters(), 0.01f);

    int num_epochs = 5;
    int batch_size = 64;
    int num_batches = (train_data.size() + batch_size - 1) / batch_size;

    std::vector<float> train_loss_history;
    std::vector<float> train_acc_history;
    std::vector<float> test_acc_history;

    for (int epoch = 0; epoch < num_epochs; epoch++) {
        float epoch_loss = 0.0f;
        int correct = 0;
        int total = 0;

        indicators::show_console_cursor(false);
        indicators::ProgressBar bar{
            indicators::option::BarWidth{50},
            indicators::option::Start{"["},
            indicators::option::Fill{"="},
            indicators::option::Lead{">"},
            indicators::option::Remainder{" "},
            indicators::option::End{"]"},
            indicators::option::PrefixText{"Epoch " + std::to_string(epoch + 1) + "/" + std::to_string(num_epochs)},
            indicators::option::ForegroundColor{indicators::Color::green},
            indicators::option::ShowPercentage{true},
            indicators::option::MaxProgress{num_batches}
        };

        for (int batch = 0; batch < num_batches; batch++) {
            auto [inputs, targets] = train_data.get_batch(batch_size, batch * batch_size);

            auto outputs = model.forward(inputs);
            auto log_probs = outputs->log_softmax();
            auto loss = log_probs->nll_loss(targets);

            optimizer.zero_grad();
            loss->backward();
            optimizer.step();

            epoch_loss += loss->data()(0, 0);

            const Eigen::MatrixXf& probs = outputs->data();
            for (int i = 0; i < targets.size(); i++) {
                Eigen::MatrixXf::Index max_row;
                probs.col(i).maxCoeff(&max_row);
                if (max_row == targets[i]) correct++;
                total++;
            }

            bar.set_option(indicators::option::PostfixText{"loss: " + std::to_string(loss->data()(0, 0))});
            bar.tick();
        }

        indicators::show_console_cursor(true);

        float train_accuracy = 100.0f * correct / total;
        std::cout << "Epoch " << epoch + 1 << "/" << num_epochs
            << ", Loss: " << epoch_loss / num_batches
            << ", Train Accuracy: " << train_accuracy << "%" << std::endl;

        train_loss_history.push_back(epoch_loss / num_batches);
        train_acc_history.push_back(train_accuracy);

        float test_accuracy = test(model, "../data/mnist/t10k-images.idx3-ubyte",
                                   "../data/mnist/t10k-labels.idx1-ubyte");
        test_acc_history.push_back(test_accuracy);

        std::cout << "Test Accuracy: " << test_accuracy << "%" << std::endl;
    }

    save_metrics(train_loss_history, train_acc_history, test_acc_history);

    return 0;
}
