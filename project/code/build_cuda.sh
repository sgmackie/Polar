#!/bin/bash
# Debian: run chmod u+x build.sh if complaining about permissions
# Make sure EOL sequence is LF! Editing scripts on Windows can change that

# Stop at script errors
set -e

# Set name of the main .cpp file for building
Kernel=polar_cuda
CurDir=$(pwd)

# Set build directory
BuildDir="${CurDir}/../build/linux"

# Make build directory if it doesn't exist
if [ ! -d "${BuildDir}" ]; then
	mkdir -p ${BuildDir}
fi

# Move to build directory
pushd ${BuildDir} > /dev/null

# Set compiler arguments
CUDAFiles="${CurDir}/cuda/${Kernel}.cu"

# Set CUDA include paths
CUDAPaths="--include-path "..\..\external\CUDA_Common""

# Set compiler flags:
CompilerFlags="--lib --debug --generate-line-info"

# Set Compiler optimsation level
CompilerOpt="-O0"

# Run NVCC compiler
nvcc ${CompilerFlags} ${CUDAPaths} ${CompilerOpt} ${CUDAFiles} -o ${Kernel}.so

# Exit
popd > /dev/null