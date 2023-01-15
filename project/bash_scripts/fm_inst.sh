#!/bin/bash    
# Recursively finds all files named instrumentals.wav and moves them to working directory and
# renames them based on which top level folder they were found in
# example: in wd there is a folder called "001 - songname" and in this folder there is a file "instrumentals.wav"
# result: instrumentals.wav is moved to wd and named 001_inst.wav. The same search and move happens for all folders in wd
find . -type f -name *instrumentals.wav | while IFS= read -r f; do
    echo ${f:2:3}
    name=${f:2:3}
    #((++i))
    mv "$f" "./${name%.*}_inst.wav"
done