# Polar

- [About](#about)
- [Build](#build)
- [Acknowledegments](#acknowledgements)

### About: <a name="about"></a>

Realtime audio engine that leverages CUDA for parallel processing.

### Build: <a name="build"></a>

All platforms were built using Clang (8.0): https://releases.llvm.org/download.html
CUDA kernels built using NVCC (10.1): 

- Windows (x64):
    - VS2017: Call vcvars64.bat from the Visual Studio install directory
    - VS2017: Run build_cuda.bat
    - VS2017: Run build.bat
- Linux (x64):
    - Clang: Run build_cuda.sh
    - Clang: Run build.sh

### Acknowledegments: <a name="acknowledgements"></a>

- Hashing function:
    - xxHash: https://cyan4973.github.io/xxHash/
- File code:
    - dr_wav: https://mackron.github.io/
- Random number generator:
    - PCG: http://www.pcg-random.org/
- Open Sound Control:
    - oscpkt: http://gruntthepeon.free.fr/oscpkt/