#!/bin/bash
# Run after creating new script: chmod 700 build.sh

# Set name of the main .cpp file for building
Main=linux_polar

# Set build directory
BuildDir=../build/linux

# Make build directory if it doesn't exist
if [ ! -d "${BuildDir}" ]; then
    mkdir ${BuildDir}
fi

# Move to build directory
pushd ${BuildDir}

# Set compiler arguments
Files=../../../project/code/${Main}.cpp
Libs=

# Set compiler flags
# -g                generate debug information
# -Wall             enable all warnings
# -Werror           treat warnings as errors
# -O0               compiler optimisations level (0,1,2,3)
CompilerFlags="-g -Wall -Werror -pedantic -std=c++11 -O0"
LinkerFlags=

# Run Clang compiler
clang++ ${Files} ${CompilerFlags} ${Libs} -o ${Main} ${LinkerFlags}


# Step out of build directory
popd