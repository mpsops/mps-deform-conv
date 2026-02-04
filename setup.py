"""
Build script for mps-deform-conv.
"""

import os
import sys
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext

class ObjCppBuildExt(build_ext):
    """Custom build_ext that handles .mm (Objective-C++) files."""

    def build_extension(self, ext):
        # Add .mm to recognized source extensions
        self.compiler.src_extensions.append('.mm')

        # Store original _compile
        original_compile = self.compiler._compile

        def patched_compile(obj, src, ext, cc_args, extra_postargs, pp_opts):
            # For .mm files, use Objective-C++ compilation
            if src.endswith('.mm'):
                # Ensure we're using clang++ with ObjC++ flag
                try:
                    self.compiler.compiler_so = ['clang++'] + [
                        arg for arg in self.compiler.compiler_so[1:]
                        if arg not in ['-Wstrict-prototypes']
                    ]
                except:
                    pass
            return original_compile(obj, src, ext, cc_args, extra_postargs, pp_opts)

        self.compiler._compile = patched_compile
        super().build_extension(ext)


def get_extensions():
    import torch
    from torch.utils.cpp_extension import CppExtension

    # Get PyTorch include paths
    torch_include = torch.utils.cpp_extension.include_paths()

    ext_modules = [
        CppExtension(
            name="mps_deform_conv._C",
            sources=["mps_deform_conv/csrc/deform_conv2d_mps.mm"],
            include_dirs=torch_include,
            extra_compile_args=["-std=c++17", "-O3"],
            extra_link_args=[
                "-framework", "Metal",
                "-framework", "Foundation",
                "-framework", "MetalPerformanceShaders",
            ],
        )
    ]
    return ext_modules


setup(
    name="mps-deform-conv",
    version="0.2.2",
    description="Deformable Convolution 2D for PyTorch on Apple Silicon (MPS)",
    author="imperatormk",
    packages=["mps_deform_conv"],
    package_data={
        "mps_deform_conv": ["kernels/*.metal"],
    },
    ext_modules=get_extensions(),
    cmdclass={"build_ext": ObjCppBuildExt},
    install_requires=["torch>=2.0.0"],
    python_requires=">=3.10",
)
