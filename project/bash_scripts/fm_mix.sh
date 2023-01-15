#!/bin/bash    
find . -type f -name *.wav | while IFS= read -r f; do
    echo ${f:2:3}
    name=${f:2:3}
    #((++i))
    mv "$f" "./${name%.*}_mix.wav"
done