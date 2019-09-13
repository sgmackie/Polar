#!/bin/bash
# Debian: run chmod u+x build.sh if complaining about permissions
# Make sure EOL sequence is LF! Editing scripts on Windows can change that

# Stop at script errors
set -e

# Move to build path
pushd ../build/linux

# Create benchmark paths
Output4="/mnt/FC663B45663AFFC6/Projects/Current/Programming/Morph/project/data/Output_4"
Output25="/mnt/FC663B45663AFFC6/Projects/Current/Programming/Morph/project/data/Output_25"
Output50="/mnt/FC663B45663AFFC6/Projects/Current/Programming/Morph/project/data/Output_50"

# Delete last outputs
rm -d -f ${Output_4}
rm -d -f ${Output_25}
rm -d -f ${Output_50}

# Make build directory if it doesn't exist
if [ ! -d "${Output4}" ]; then
	mkdir -p ${Output4}
fi
if [ ! -d "${Output25}" ]; then
	mkdir -p ${Output25}
fi
if [ ! -d "${Output50}" ]; then
	mkdir -p ${Output50}
fi

# 1 Morph
if [ $1 -eq 4 ] 
then
    $2 ./morph "/mnt/FC663B45663AFFC6/Projects/Current/Programming/Morph/project/data/BOOM_A" "/mnt/FC663B45663AFFC6/Projects/Current/Programming/Morph/project/data/BOOM_B" 4 ${Output4} 1 0.5 0.9
    echo
# 5 Morph
elif [ $1 -eq 25 ] 
then
    $2 ./morph "/mnt/FC663B45663AFFC6/Projects/Current/Programming/Morph/project/data/BOOM_A" "/mnt/FC663B45663AFFC6/Projects/Current/Programming/Morph/project/data/BOOM_B" 25 ${Output25} 1 0.5 0.9
    echo
# 10 Morph    
else
    $2 ./morph "/mnt/FC663B45663AFFC6/Projects/Current/Programming/Morph/project/data/BOOM_A" "/mnt/FC663B45663AFFC6/Projects/Current/Programming/Morph/project/data/BOOM_B" 50 ${Output50} 1 0.5 0.9
    echo
fi

popd