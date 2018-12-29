#!/bin/bash
# Debian: run chmod u+x build.sh if complaining about permissions
# Make sure EOL sequence is LF! Editing scripts on Windows can change that

# Stop at script errors
set -e

# Set name of the main .cpp file for building
Platform=linux_polar
Engine=polar
CurDir=$(pwd)

# Set CTime directory relative to current drive and path
CTimeDir="${CurDir}/../build/ctime"

# Check if CTime path exists
if [ ! -d "${CTimeDir}" ]; then
	mkdir -p ${CTimeDir}
fi

# Move to CTime directory
pushd ${CTimeDir} > /dev/null

# Begin CTime on .cpp files
#./ctime_linux -begin ${Engine}.ctm
#./ctime_linux -begin ${Platform}.ctm

# Step out of CTime directory
popd > /dev/null

# Set build directory
BuildDir="${CurDir}/../build/linux"

# Make build directory if it doesn't exist
if [ ! -d "${BuildDir}" ]; then
	mkdir -p ${BuildDir}
fi

# Move to build directory
pushd ${BuildDir} > /dev/null

# Set compiler arguments
EngineFiles="${CurDir}/${Engine}.cpp"
PlatformFiles="${CurDir}/${Platform}.cpp"

# Set Clang compiler flags (https://clang.llvm.org/docs/genindex.html)
# -g                generate debug information
# -Wall             enable all warnings
# -Werror           treat warnings as errors
# -Wno              to disable Clang warnings:  unused-function
# -pedantic         warn about language extensions
# -std=c++11        choose C++ 2011 Standard if linking to the C++ library
CompilerFlags="-g -Wall -Werror -Wno-unused-function -pedantic -std=c++11"

# Set Compiler optimsation level for debug or release builds
# -O0               compiler optimisations level
CompilerOpt="-O0"

# Set Linux libraries
# -lm               CRT math library
# -ldl              dynamic link library support
# -lX11             X11 Unix windowing system
# -lasound          ALSA library
Libs="-lm -ldl -lX11 -lasound"

# Set link flags:
# -pthread          enable POSIX threads
LinkerFlags="-pthread"

# Run Clang compiler
# Polar:
# -shared           to create shared object library for dynamic linking
# -fPIC             Position Independant Code: required by shared
clang ${CompilerFlags} -shared -fPIC ${CompilerOpt} ${Libs} ${EngineFiles} -o ${Engine}.so ${LinkerFlags}
PolarLastError=${ERRORLEVEL}

# Linux:
clang ${CompilerFlags} ${CompilerOpt} ${Libs} ${PlatformFiles} -o ${Platform} ${LinkerFlags}
PlatformLastError=${ERRORLEVEL}

# Step out of build directory
popd > /dev/null

# Move back to CTime path
pushd ${CTimeDir} > /dev/null

# End CTime on .cpp files
#./ctime_linux -end ${Engine}.ctm ${PolarLastError}
#./ctime_linux -end ${Platform}.ctm ${PlatformLastError}

# Exit
popd > /dev/null
