#!/bin/bash
# Debian: run chmod u+x program_name if complaining about permissions
# Make sure EOL sequence is LF! Editing scripts on Windows can change that

# Stop at script errors
set -e

# Set name of the main .cpp file for building
Platform=linux_polar
CurDir=$(pwd)

# Set CTime directory relative to current drive and path
CTimeDir="${CurDir}/../build/ctime"

# Move to CTime directory
pushd ${CTimeDir} > /dev/null

# Begin CTime on .cpp files
./ctime_linux -begin ${Platform}.ctm

# Step out of CTime directory
popd > /dev/null

# Set build directory
BuildDir="${CurDir}/../build/linux"

# Make build directory if it doesn't exist
mkdir -p ${BuildDir}

# Move to build directory
pushd ${BuildDir} > /dev/null

# Set compiler arguments
PlatformFiles=../../../project/code/${Platform}.cpp

# Set Clang compiler flags (https://clang.llvm.org/docs/genindex.html)
# -g                generate debug information
# -Wall             enable all warnings
# -Werror           treat warnings as errors
# -pedantic         warn about language extensions
# -std=c++11        choose C++ 2011 Standard if linking to the C++ library
CompilerFlags="-g -Wall -Werror -pedantic"

# Set Compiler optimsation level for debug or release builds
# -O0               compiler optimisations level
CompilerOpt="-O0"

# Set Linux libraries
# -lasound          ALSA library
Libs=-lasound

# Set link flags:
# -pthread          enable POSIX threads
LinkerFlags="-pthread -lX11 -ldl"

# Run Clang compiler
# Linux:
clang ${PlatformFiles} ${CompilerFlags} ${CompilerOpt} ${Libs} -o ${Platform} #${LinkerFlags}
PlatformLastError=${ERRORLEVEL}

# Step out of build directory
popd > /dev/null

# Move back to CTime path
pushd ${CTimeDir} > /dev/null

# End CTime on .cpp files
./ctime_linux -end ${Platform}.ctm ${PlatformLastError}

# Exit
popd > /dev/null