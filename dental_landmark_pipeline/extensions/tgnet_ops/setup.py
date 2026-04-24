# Build TGNet CUDA ops as "tgnet_ops"
# Targets Ampere sm_86 (RTX 3050 Laptop) + CUDA 12.6 / PyTorch 2.10
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='tgnet_ops',
    author='Hengshuang Zhao (adapted for dental_landmark_pipeline)',
    ext_modules=[
        CUDAExtension('tgnet_ops', [
            'src/pointops_api.cpp',
            'src/knnquery/knnquery_cuda.cpp',
            'src/knnquery/knnquery_cuda_kernel.cu',
            'src/sampling/sampling_cuda.cpp',
            'src/sampling/sampling_cuda_kernel.cu',
            'src/grouping/grouping_cuda.cpp',
            'src/grouping/grouping_cuda_kernel.cu',
            'src/interpolation/interpolation_cuda.cpp',
            'src/interpolation/interpolation_cuda_kernel.cu',
            'src/subtraction/subtraction_cuda.cpp',
            'src/subtraction/subtraction_cuda_kernel.cu',
            'src/aggregation/aggregation_cuda.cpp',
            'src/aggregation/aggregation_cuda_kernel.cu',
        ],
        extra_compile_args={
            'cxx': ['/O2'],
            'nvcc': [
                '-O2',
                '-gencode', 'arch=compute_86,code=sm_86',
                '-allow-unsupported-compiler',   # VS 2026 not officially supported by CUDA 12.6
            ]
        })
    ],
    cmdclass={'build_ext': BuildExtension}
)
