#include <torch/torch.h>
#include <torch/script.h>
#include <torch/data/datasets/mnist.h>
#include <torch/data/transforms.h>
#include <iostream>
#include <vector>

class Net : public torch::nn::Module {
public:
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

void train(int epoch, int log_interval = 200) {
    model->train();

    for (const auto& batch : *train_loader) {
        auto data = batch.data.to(device);
        auto target = batch.target.to(device);

        optimizer.zero_grad();
        auto output = model->forward(data);
        auto loss = criterion->forward(output, target);
        loss.backward();
        optimizer.step();

        if (batch.index() % log_interval == 0) {
            std::cout << "Train Epoch: " << epoch << " ["
                << batch.index() * data.size(0) << "/"
                << train_loader->size().value() << " ("
                << 100. * batch.index() / train_loader->size().value()
                << "%)]\tLoss: " << loss.item<float>() << std::endl;
        }
    }
}

void validate(std::vector<float>& loss_vector, std::vector<float>& accuracy_vector) {
    model->eval();
    float val_loss = 0;
    int correct = 0;

    for (const auto& batch : *validation_loader) {
        auto data = batch.data.to(device);
        auto target = batch.target.to(device);
        auto output = model->forward(data);
        val_loss += criterion->forward(output, target).item<float>();
        auto pred = output.argmax(1); // get the index of the max log-probability
        correct += pred.eq(target).sum().item<int>();
    }

    val_loss /= validation_loader->size().value();
    loss_vector.push_back(val_loss);

    float accuracy = 100. * static_cast<float>(correct) / validation_loader->size().value();
    accuracy_vector.push_back(accuracy);

    std::cout << "\nValidation set: Average loss: " << val_loss
        << ", Accuracy: " << correct << "/" << validation_loader->size().value()
        << " (" << accuracy << "%)\n" << std::endl;
}

int main() {
    torch::manual_seed(1);

    torch::DeviceType device_type;
    if (torch::cuda::is_available()) {
        std::cout << "CUDA available. Using GPU." << std::endl;
        device_type = torch::kCUDA;
    }
    else {
        std::cout << "CUDA not available. Using CPU." << std::endl;
        device_type = torch::kCPU;
    }
    torch::Device device(device_type);

    const int64_t batch_size = 32;

    auto train_dataset = torch::data::datasets::MNIST(
        "./data",
        torch::data::datasets::MNIST::Mode::kTrain,
        torch::data::datasets::MNIST::kNone,
        torch::data::datasets::MNIST::kNone,
        torch::data::datasets::MNIST::kNone);
    auto train_loader = torch::data::make_data_loader(
        std::move(train_dataset), batch_size);

    auto validation_dataset = torch::data::datasets::MNIST(
        "./data",
        torch::data::datasets::MNIST::Mode::kTest,
        torch::data::datasets::MNIST::kNone,
        torch::data::datasets::MNIST::kNone,
        torch::data::datasets::MNIST::kNone);
    auto validation_loader = torch::data::make_data_loader(
        std::move(validation_dataset), batch_size);

    for (const auto& batch : *train_loader) {
        const auto& images = batch.data.to(device);
        const auto& labels = batch.target.to(device);
        std::cout << "images: " << images.sizes() << " type: "
            << images.scalar_type() << std::endl;
        std::cout << "labels: " << labels.sizes() << " type: "
            << labels.scalar_type() << std::endl;
        break;
    }

    Net model;
    model.to(device);

    torch::optim::SGD optimizer(
        model.parameters(), torch::optim::SGDOptions(0.01).momentum(0.5));
    torch::optim::StepLR scheduler(optimizer, 3, 0.1);
    torch::nn::CrossEntropyLoss criterion;

    std::cout << model << std::endl;

    int epochs = 10;
    std::vector<float> lossv, accv;

    for (int epoch = 1; epoch <= epochs; ++epoch) {
        train(epoch);
        validate(lossv, accv);
    }

    torch::save(model->state_dict(), "../../Srcs/mnist_trained_model_v3.pth");

    return 0;
}