#!/bin/bash
# Run after creating new script: chmod 700 build.sh
# Debian: run chmod u+x program_name if complaining about permissions
# Make sure EOL sequence is LF!

# Stop at script errors
set -e

# Set name of the main .cpp file for building
Main=mac_polar

# Begin CTime on .cpp files
./ctime_mac -begin ${Main}.ctm

# Set build directory
CurDir=$(pwd)
BuildDir="${CurDir}/../build/mac"

# Make build directory if it doesn't exist
mkdir -p ${BuildDir}

# Move to build directory
pushd ${BuildDir} > /dev/null

# Set compiler arguments
Files=../../../project/code/${Main}.cpp
Libs=

# Set Clang compiler flags (https://clang.llvm.org/docs/genindex.html)
# -g                generate debug information
# -Wall             enable all warnings
# -Werror           treat warnings as errors
# -pedantic         warn about language extensions
# -std=c++11        choose C++ standard (2011)
# -O0               compiler optimisations level
CompilerFlags="-g -Wall -Werror -pedantic -std=c++11 -O0"

# Set link flags:
# -pthread          enable POSIX threads
LinkerFlags="-pthread -lX11 -ldl"

# Run Clang compiler
clang++ ${Files} ${CompilerFlags} ${Libs} -o ${Main} #${LinkerFlags}

# Step out of build directory
popd > /dev/null

# End CTime on .cpp files
./ctime_mac -end ${Main}.ctm