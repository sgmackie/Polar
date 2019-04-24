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

# Set compiler flags:
CompilerFlags="-g -gcodeview -pedantic"

# Set warning labels:
CommonWarnings="-Wall -Werror -Wno-language-extension-token -Wno-deprecated-declarations -Wno-unused-variable -Wno-unused-function"

# Set Compiler optimsation level
CompilerOpt="-O0"

# Set logging flags
LogFlags="-DLOGGER_ERROR=0"

# Set profile flags
ProfileFlags="-DMICROPROFILE=0 -DMICROPROFILE_UI=0 -DMICROPROFILE_WEBSERVER=1 -DMICROPROFILE_GPU_TIMERS=0"

# Set Linux libraries
Libs="-lm -lasound"

# Run Clang compiler
clang++ ${CompilerFlags} ${CommonWarnings} ${CompilerOpt} ${LogFlags} ${ProfileFlags} ${Libs} ${PlatformFiles} -o ${Platform}

# Exit
popd > /dev/null