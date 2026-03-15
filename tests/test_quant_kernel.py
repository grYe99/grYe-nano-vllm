import pytest
import torch
from nanovllm.layers.quant_kernel import w4a16_gemm, pack_weight_to_int4


def make_test_inputs(M=4, K=256, N=128, device="cuda"):
    torch.manual_seed(42)
    x = torch.randn(M, K, dtype=torch.float16, device=device)
    w = torch.randn(N, K, dtype=torch.float16, device=device)
    return x, w


@pytest.mark.skipif(not torch.cuda.is_available(), reason="需要 GPU")
def test_pack_unpack_roundtrip():
    _, w = make_test_inputs()
    w_packed, scales, zeros = pack_weight_to_int4(w)
    N, K = w.shape
    group_size = 128
    assert w_packed.shape == (K // 2, N)
    assert scales.shape == (K // group_size, N)
    assert zeros.shape == (K // group_size, N)
    assert w_packed.dtype == torch.uint8
    assert scales.dtype == torch.float16


@pytest.mark.skipif(not torch.cuda.is_available(), reason="需要 GPU")
def test_w4a16_gemm_correctness():
    M, K, N = 4, 256, 128
    x, w = make_test_inputs(M, K, N)
    w_packed, scales, zeros = pack_weight_to_int4(w, group_size=128)

    # 参考：手动 dequant 再矩阵乘
    num_groups = K // 128
    w_T = w.T.float().reshape(num_groups, 128, N)
    w_min = w_T.amin(dim=1, keepdim=True)
    w_max = w_T.amax(dim=1, keepdim=True)
    s = ((w_max - w_min) / 15.0).clamp(min=1e-8)
    w_quant = ((w_T - w_min) / s).round().clamp(0, 15)
    w_dequant = (w_quant * s + w_min).reshape(K, N).T.half()  # [N, K]
    ref_out = x @ w_dequant.T

    fused_out = w4a16_gemm(x, w_packed, scales, zeros, group_size=128)

    assert fused_out.shape == (M, N)
    max_err = (fused_out.float() - ref_out.float()).abs().max().item()
    # int4 quantization error bound: ~scale/2, typically 0.05-0.15 for randn weights
    assert max_err < 0.15, f"max error {max_err:.4f} exceeds int4 quantization tolerance"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="需要 GPU")
def test_w4a16_gemm_large():
    M, K, N = 1, 1024, 1024
    x = torch.randn(M, K, dtype=torch.float16, device="cuda")
    w = torch.randn(N, K, dtype=torch.float16, device="cuda")
    w_packed, scales, zeros = pack_weight_to_int4(w)
    out = w4a16_gemm(x, w_packed, scales, zeros)
    assert out.shape == (M, N)
    assert not out.isnan().any()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="需要 GPU")
def test_w4a16_linear_forward():
    """W4A16Linear forward 应与 dequant 参考实现误差在量化范围内"""
    from nanovllm.layers.linear import W4A16Linear

    torch.manual_seed(42)
    M, K, N = 4, 256, 128
    group_size = 128

    fp16_weight = torch.randn(N, K, dtype=torch.float16, device="cuda")
    layer = W4A16Linear(input_size=K, output_size=N, group_size=group_size)
    layer.to("cuda")
    layer.pack_weights(fp16_weight)

    x = torch.randn(M, K, dtype=torch.float16, device="cuda")

    # 参考：手动 dequant 再矩阵乘（与 test_w4a16_gemm_correctness 一致）
    num_groups = K // group_size
    w_T = fp16_weight.T.float().reshape(num_groups, group_size, N)
    w_min = w_T.amin(dim=1, keepdim=True)
    w_max = w_T.amax(dim=1, keepdim=True)
    s = ((w_max - w_min) / 15.0).clamp(min=1e-8)
    w_quant = ((w_T - w_min) / s).round().clamp(0, 15)
    w_dequant = (w_quant * s + w_min).reshape(K, N).T.half()  # [N, K]
    ref = x @ w_dequant.T

    out = layer(x)

    assert out.shape == (M, N)
    assert not out.isnan().any()
    # 与 dequant 参考对齐，允许小数值误差（相对误差 < 5%）
    rel_err = (out.float() - ref.float()).abs() / (ref.float().abs() + 1e-6)
    assert rel_err.mean().item() < 0.05, f"mean rel error {rel_err.mean().item():.4f} too large"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="需要 GPU")
def test_quantize_model_replaces_layers():
    """quantize_model 应将 ColumnParallel/RowParallel 替换为 W4A16 变体"""
    import torch.distributed as dist
    if not dist.is_initialized():
        import tempfile, os
        store = dist.FileStore(os.path.join(tempfile.gettempdir(), f"test_qm_{os.getpid()}"), 1)
        dist.init_process_group("gloo", store=store, rank=0, world_size=1)

    from nanovllm.layers.linear import (
        ColumnParallelLinear, RowParallelLinear,
        W4A16ColumnParallelLinear, W4A16RowParallelLinear,
        quantize_model,
    )
    import torch.nn as nn

    torch.set_default_device("cuda")

    class TinyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = ColumnParallelLinear(128, 256)
            self.fc2 = RowParallelLinear(256, 128)

        def forward(self, x):
            return self.fc2(self.fc1(x))

    model = TinyModel()
    nn.init.normal_(model.fc1.weight)
    nn.init.normal_(model.fc2.weight)

    quantize_model(model, group_size=128)

    assert isinstance(model.fc1, W4A16ColumnParallelLinear), \
        f"Expected W4A16ColumnParallelLinear, got {type(model.fc1)}"
    assert isinstance(model.fc2, W4A16RowParallelLinear), \
        f"Expected W4A16RowParallelLinear, got {type(model.fc2)}"

    # forward 应正常工作，无 NaN
    x = torch.randn(2, 128, dtype=torch.float16, device="cuda")
    out = model(x)
    assert out.shape == (2, 128)
    assert not out.isnan().any()
    torch.set_default_device("cpu")
