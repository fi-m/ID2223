#!/bin/bash -e
shopt -s globstar
# for every folder in currend working directory, run ffmpeg amix on the drums, bass and other files in that directory
for f in ./*/; do
    echo "$f"
    d="${f}drums.wav" 
    b="${f}bass.wav" 
    o="${f}other.wav" 
     
    # echo "d:$d" # debug
    # echo "b:$b" # debug
    # echo "o:$o" # debug
    
    ffmpeg -i "$d" -i "$b" -i "$o" -filter_complex amix=inputs=3:duration=first:dropout_transition=3 "${f}instrumentals.wav"
    
done
