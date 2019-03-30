# Polar

- [About](#about)
- [Build](#build)
- [Acknowledgments](#acknowledgments)

### About: <a name="about"></a>

Realtime audio engine that leverages CUDA for parallel processing.

### Build: <a name="build"></a>

All platforms were built using Clang (8.0): https://releases.llvm.org/download.html

CUDA kernels built using NVCC (10.1):       https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/

- Windows (x64):
    - VS2017: Call vcvars64.bat from the Visual Studio install directory
    - NVCC: Run build_cuda.bat
    - Clang: Run build.bat
- Linux (x64):
    - NVCC: Run build_cuda.sh
    - Clang: Run build.sh

### Acknowledgments: <a name="acknowledgments"></a>

- Documentation:
    - stddoc.c:     https://github.com/r-lyeh/stddoc.c
- IMGUI:
    - Dear ImGui:   https://github.com/ocornut/imgui
- Hashing function:
    - xxHash:       https://cyan4973.github.io/xxHash/
- File code:
    - dr_wav:       https://mackron.github.io/
- Random number generator:
    - PCG:          http://www.pcg-random.org/
- Open Sound Control:
    - oscpkt:       http://gruntthepeon.free.fr/oscpkt/
- CUDA Errors:
    - CUDA SDK:     https://docs.nvidia.com/cuda/cuda-samples/index.html