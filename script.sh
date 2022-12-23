#!/bin/bash

loc="/Volumes/Extreme/Projects/staging/raw/*"
for i in $loc
do
    echo $i
    cd bftools
    file="$i.txt"
    ./showinf -nopix $i > $file
    cd ..
done