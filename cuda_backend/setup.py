from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


setup(
    name="paged_attention_cuda",
    ext_modules=[
        CUDAExtension(
            name="paged_attention_cuda",
            sources=[
                "paged_attention.cpp",
                "paged_attention_kernel.cu",
            ],
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": ["-O3"],
            },
        )
    ],
    cmdclass={
        "build_ext": BuildExtension,
    },
)