// Torch bindings for AWQ custom ops.
// Separate .cpp file so TORCH_LIBRARY static initializers work correctly
// (they may not fire in nvcc-compiled .cu files).

#include <torch/extension.h>

// Forward-declare the host wrapper defined in gemm_kernels.cu
torch::Tensor awq_gemm(torch::Tensor _in_feats,
                       torch::Tensor _kernel,
                       torch::Tensor _scaling_factors,
                       torch::Tensor _zeros,
                       int64_t split_k_iters);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("awq_gemm", &awq_gemm, "AWQ quantized GEMM forward");
}

TORCH_LIBRARY(nanovllm, m) {
    m.def("awq_gemm(Tensor in_feats, Tensor kernel, "
           "Tensor scaling_factors, Tensor zeros, "
           "int split_k_iters) -> Tensor");
}

TORCH_LIBRARY_IMPL(nanovllm, CUDA, m) {
    m.impl("awq_gemm", &awq_gemm);
}
