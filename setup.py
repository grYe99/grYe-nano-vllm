from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension
import os

setup(
    ext_modules=[
        CUDAExtension(
            "nanovllm._C",
            ["nanovllm/csrc/awq/gemm_kernels.cu",
             "nanovllm/csrc/awq/torch_bindings.cpp"],
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": [
                    "-O3",
                    "-std=c++17",
                    "--generate-code=arch=compute_89,code=sm_89",
                ],
            },
        ),
        CUDAExtension(
            "nanovllm._C_marlin",
            ["nanovllm/csrc/marlin/marlin.cu",
             "nanovllm/csrc/marlin/awq_marlin_repack.cu",
             "nanovllm/csrc/marlin/sm80_kernel_float16_u4_float16.cu",
             "nanovllm/csrc/marlin/torch_bindings_marlin.cpp"],
            extra_compile_args={
                "cxx": ["-O3", "-std=c++17",
                        "-I%s/nanovllm/csrc" % os.path.dirname(os.path.abspath(__file__))],
                "nvcc": [
                    "-O3",
                    "-std=c++17",
                    "--generate-code=arch=compute_89,code=sm_89",
                    "-DMARLIN_NAMESPACE_NAME=nanovllm_marlin",
                    "--use_fast_math",
                    "-I%s/nanovllm/csrc" % os.path.dirname(os.path.abspath(__file__)),
                ],
            },
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
)
