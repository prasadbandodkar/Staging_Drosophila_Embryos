#!/bin/bash

# script to get metadata from all files in a directory using showinf from bftools provided by bftools
# bftools can be downloaded from https://bio-formats.readthedocs.io/en/stable/users/comlinetools/index.html

loc="/Volumes/Extreme/Projects/staging/raw/*"
for i in $loc
do
    echo $i
    cd bftools
    file="$i.txt"
    ./showinf -nopix $i > $file
    cd ..
done