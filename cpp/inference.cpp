#include <torch/script.h> // One-stop header for including PyTorch C++ API
#include <pybind11/pybind11.h>
#include <iostream>

namespace py = pybind11;

torch::jit::script::Module load_model(const std::string &model_path) {
    // Deserialize the ScriptModule from a file using torch::jit::load().
    torch::jit::script::Module module;
    try {
        module = torch::jit::load(model_path);
    }
    catch (const c10::Error &e) {
        std::cerr << "Error loading the model\n";
        exit(EXIT_FAILURE);
    }
    return module;
}

int infer(const torch::jit::script::Module &model, const torch::Tensor &input) {
    torch::NoGradGuard no_grad; // Ensure that autograd is off
    model.eval(); // Set to evaluation mode
    auto output = model.forward({input}).toTensor();
    auto pred = output.argmax(1).item<int>();
    return pred;
}

PYBIND11_MODULE(inference, m) {
    m.def("load_model", &load_model, "A function that loads a PyTorch model");
    m.def("infer", &infer, "A function that performs inference");
}

