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
    // Marlin ops (implementations in _C_marlin extension):
    m.def("marlin_gemm(Tensor a, Tensor? c, Tensor b_q_weight, Tensor? bias, "
           "Tensor b_scales, Tensor? a_scales, Tensor? global_scale, "
           "Tensor? b_zeros, Tensor? g_idx, Tensor? perm, Tensor workspace, "
           "int b_type_id, int size_m, int size_n, int size_k, "
           "bool is_k_full, bool use_atomic_add, bool use_fp32_reduce, "
           "bool is_zp_float) -> Tensor");
    m.def("awq_marlin_repack(Tensor b_q_weight, int size_k, int size_n, "
           "int num_bits, bool sym) -> Tensor");
}

TORCH_LIBRARY_IMPL(nanovllm, CUDA, m) {
    m.impl("awq_gemm", &awq_gemm);
}
