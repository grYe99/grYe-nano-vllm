// Torch bindings for Marlin AWQ ops.
// Separate .cpp file so TORCH_LIBRARY static initializers work correctly
// (they may not fire in nvcc-compiled .cu files).

#include <torch/extension.h>
#include "core/registration.h"

// ---- Forward declarations from marlin.cu and awq_marlin_repack.cu ----

torch::Tensor marlin_gemm(
    torch::Tensor& a, std::optional<torch::Tensor> c_or_none,
    torch::Tensor& b_q_weight,
    std::optional<torch::Tensor> const& b_bias_or_none,
    torch::Tensor& b_scales,
    std::optional<torch::Tensor> const& a_scales_or_none,
    std::optional<torch::Tensor> const& global_scale_or_none,
    std::optional<torch::Tensor> const& b_zeros_or_none,
    std::optional<torch::Tensor> const& g_idx_or_none,
    std::optional<torch::Tensor> const& perm_or_none,
    torch::Tensor& workspace,
    int64_t const& b_type_id,
    int64_t size_m, int64_t size_n, int64_t size_k,
    bool is_k_full, bool use_atomic_add, bool use_fp32_reduce,
    bool is_zp_float);

torch::Tensor awq_marlin_repack(torch::Tensor& b_q_weight, int64_t size_k,
                                int64_t size_n, int64_t num_bits, bool sym);

// ---- PYBIND11 module (provides PyInit) ----
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("marlin_gemm", &marlin_gemm, "Marlin quantized GEMM forward");
  m.def("awq_marlin_repack", &awq_marlin_repack,
        "Repack AWQ int4 weights to Marlin format");
}

// Op schemas are defined in torch_bindings.cpp (nanovllm TORCH_LIBRARY).
// Here we just register CUDA implementations.
TORCH_LIBRARY_IMPL(nanovllm, CUDA, m) {
  m.impl("marlin_gemm", &marlin_gemm);
  m.impl("awq_marlin_repack", &awq_marlin_repack);
}
