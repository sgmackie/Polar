echo NVCC x64

@echo off

:: Set name of the kernel .cu file for unity building
set Kernel=polar_cuda

:: Set build directory relative to current drive and path
set BuildDir="%~dp0..\build\win32"

:: Create build path if it doesn't exist
if not exist %BuildDir% mkdir %BuildDir%

:: Move to build directory
pushd %BuildDir%

:: Set compiler arguments
set CUDAFiles="%~dp0\cuda\%Kernel%.cu"

:: Set CUDA include paths
set CUDAPaths=--include-path "..\..\external\CUDA_Common"

:: Set compiler flags:
set CompilerFlags=--lib --debug --generate-line-info -DCUDA=1 -DCORE_PROFILE=1 -DCUV2=1 -DCUV3=0

:: Set Compiler optimsation level
REM set CompilerOpt=-O0
set CompilerOpt=-O3 --gpu-architecture=sm_52

:: Run NVCC compiler
nvcc %CompilerFlags% %CUDAPaths% %CompilerOpt% %CUDAFiles% -o %Kernel%.lib

:: Exit
popd