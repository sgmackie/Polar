#!/bin/bash
# Debian: run chmod u+x build.sh if complaining about permissions
# Make sure EOL sequence is LF! Editing scripts on Windows can change that

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

# Set Clang compiler flags (https://clang.llvm.org/docs/genindex.html)
CompilerFlags="-g -pedantic -std=c++14 -DCUDA=1"

# Set warning labels:
CommonWarnings="-Wall -Werror -Wno-unused-function -Wno-vla-extension"

# Set Compiler optimsation level for debug or release builds
CompilerOpt="-O0"

# Set Linux libraries
Libs="-lm -lasound"

# Run Clang compiler
clang ${CompilerFlags} ${CompilerWarnings} ${CompilerOpt} ${Libs} ${PlatformFiles} -o ${Platform}

# Exit
popd > /dev/null
