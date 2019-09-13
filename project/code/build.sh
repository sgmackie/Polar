#!/bin/bash
# Debian: run chmod u+x build.sh if complaining about permissions
# Make sure EOL sequence is LF! Editing scripts on Windows can change that

# ./build_cuda.sh

# Stop at script errors
set -e

# Set name of the main .cpp file for building
Platform=linux_polar
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
PlatformFiles="${CurDir}/${Platform}.cpp"

# Set CUDA include paths
CUDAPaths="-I/usr/local/cuda/include -L/usr/local/cuda/lib -L${BuildDir}"

# Set compiler flags:
CompilerFlags="-g -gcodeview -pedantic -std=c++11 `pkg-config --cflags glfw3`"

# Set warning labels:
CommonWarnings="-Wall -Werror -Wno-language-extension-token -Wno-deprecated-declarations -Wno-unused-variable -Wno-unused-function -Wno-writable-strings -Wno-gnu-anonymous-struct -Wno-variadic-macros -Wno-c++11-long-long -Wno-c++11-extensions -Wno-newline-eof"

# Set Compiler optimsation level
CompilerOpt="-O0"
# CompilerOpt="-O3 -march=native"

# Set logging flags
LogFlags="-DLOGGER_ERROR=0"

# Set CUDA flags
CUDAFlags="-DCUDA=0 -DPARTIALS_GPU=0 -DBUBBLES_GPU=0"

# Set logging flags
LogFlags="-DLOGGER_ERROR=1 -DLOGGER_PROFILE=0"

# Set profile flags
ProfileFlags="-DCORE_PROFILE=1"

# Set Linux libraries
Libs="-lm -lasound -lGLEW -lGL `pkg-config --static --libs glfw3`"

# Set path for CUDA function library
# Path links
#ln -s /usr/local/cuda/lib64/libcudart.so /usr/lib/libcudart.so
#ln -s ${BuildDir}/polar_cuda.so /usr/lib/polar_cuda.so
# CUDAFunctions="-lcudart"
CUDAFunctions=

# Run Clang compiler
clang++ ${CompilerFlags} ${CUDAPaths} ${CommonWarnings} ${CompilerOpt} ${CUDAFlags} ${LogFlags} ${ProfileFlags} ${Libs} ${CUDAFunctions} ${PlatformFiles} polar_cuda.so -o ${Platform}

# Exit
popd > /dev/null