#include <torch/torch.h>
#include <torch/script.h>
#include <torch/data/datasets/mnist.h>
#include <torch/data/transforms.h>
#include <torch/serialize.h>
#include <iostream>
#include <vector>
#include <numeric>

using namespace std;
using namespace torch;

int main() {
    torch::Device device(torch::kCPU); // Start with CPU
    if (torch::cuda::is_available()) {
        std::cout << "CUDA is available! But we're sticking to CPU.\n"; // Let's go with friendly CPU vibes!
    }
    else {
        std::cout << "CUDA not available. Using CPU.\n"; // CPU all the way!
    }

    auto train_dataset = torch::data::datasets::MNIST("./data")
        .map(torch::data::transforms::Normalize<>(0.5, 0.5))
        .map(torch::data::transforms::Stack<>());
    auto train_loader = torch::data::make_data_loader(
        std::move(train_dataset), torch::data::DataLoaderOptions().batch_size(64).workers(2));

    auto test_dataset = torch::data::datasets::MNIST("./data", torch::data::datasets::MNIST::Mode::kTest)
        .map(torch::data::transforms::Normalize<>(0.5, 0.5))
        .map(torch::data::transforms::Stack<>());
    auto test_loader = torch::data::make_data_loader(
        std::move(test_dataset), torch::data::DataLoaderOptions().batch_size(64).workers(2));

    struct Net : torch::nn::Module {
        Net() {
            conv1 = register_module("conv1", torch::nn::Conv2d(torch::nn::Conv2dOptions(1, 32, 3)));
            conv1_bn = register_module("conv1_bn", torch::nn::BatchNorm2d(32));
            conv2 = register_module("conv2", torch::nn::Conv2d(torch::nn::Conv2dOptions(32, 64, 3)));
            conv2_bn = register_module("conv2_bn", torch::nn::BatchNorm2d(64));
            conv3 = register_module("conv3", torch::nn::Conv2d(torch::nn::Conv2dOptions(64, 128, 3)));
            conv3_bn = register_module("conv3_bn", torch::nn::BatchNorm2d(128));
            fc1 = register_module("fc1", torch::nn::Linear(128 * 5 * 5, 256));
            fc1_bn = register_module("fc1_bn", torch::nn::BatchNorm1d(256));
            dropout = register_module("dropout", torch::nn::Dropout(0.5));
            fc2 = register_module("fc2", torch::nn::Linear(256, 128));
            fc2_bn = register_module("fc2_bn", torch::nn::BatchNorm1d(128));
            fc3 = register_module("fc3", torch::nn::Linear(128, 10));
        }

        torch::Tensor forward(torch::Tensor x) {
            x = conv1_bn->forward(conv1->forward(x));
            x = torch::relu(x);
            x = conv2_bn->forward(conv2->forward(x));
            x = torch::relu(x);
            x = torch::max_pool2d(x, 2);
            x = conv3_bn->forward(conv3->forward(x));
            x = torch::relu(x);
            x = torch::max_pool2d(x, 2);
            x = x.view({ -1, 128 * 5 * 5 });
            x = fc1_bn->forward(fc1->forward(x));
            x = torch::relu(x);
            x = dropout->forward(x);
            x = fc2_bn->forward(fc2->forward(x));
            x = torch::relu(x);
            x = dropout->forward(x);
            x = fc3->forward(x);
            return x;
        }

        torch::nn::Conv2d conv1{nullptr};
        torch::nn::BatchNorm2d conv1_bn{nullptr};
        torch::nn::Conv2d conv2{nullptr};
        torch::nn::BatchNorm2d conv2_bn{nullptr};
        torch::nn::Conv2d conv3{nullptr};
        torch::nn::BatchNorm2d conv3_bn{nullptr};
        torch::nn::Linear fc1{nullptr};
        torch::nn::BatchNorm1d fc1_bn{nullptr};
        torch::nn::Dropout dropout{nullptr};
        torch::nn::Linear fc2{nullptr};
        torch::nn::BatchNorm1d fc2_bn{nullptr};
        torch::nn::Linear fc3{nullptr};
    };

    Net net;
    net.to(device);

    torch::nn::CrossEntropyLoss criterion;
    torch::optim::SGD optimizer(net.parameters(), torch::optim::SGDOptions(0.01));

    // Training loop
    for (size_t epoch = 0; epoch < 11; ++epoch) {
        net.train();
        for (auto& batch : *train_loader) {
            auto data = batch.data.to(device);
            auto targets = batch.target.to(device);

            optimizer.zero_grad();
            auto outputs = net.forward(data);
            auto loss = criterion(outputs, targets);
            loss.backward();
            optimizer.step();
        }

        // Validation loop
        net.eval();
        float correct = 0;
        float total = 0;
        for (const auto& batch : *test_loader) {
            auto data = batch.data.to(device);
            auto targets = batch.target.to(device);

            auto outputs = net.forward(data);
            auto predicted = torch::argmax(outputs, 1);  // Use argmax to get predicted labels

            // Compare predicted labels with actual labels
            auto correct_predictions = predicted.eq(targets);
            correct += correct_predictions.sum().template item<float>();
        }

        // Calculate accuracy
        float accuracy = (correct / total) * 100;
        std::cout << "Epoch " << epoch + 1 << " - Validation Accuracy: " << accuracy << "%\n";
    }

    std::cout << "Finished training.\n";

    // Save the trained model
    torch::serialize::OutputArchive archive;
    net.save(archive);
    archive.save_to("D:/Internship/trained model/MNIST Trained model in c++.pt");
    std::cout << "Trained model saved.\n";


    return 0;
}