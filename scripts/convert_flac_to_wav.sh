#!/bin/bash

for flac_file in $(find . -name *.flac) ; do
  ffmpeg -hide_banner -i "$flac_file" -ar 16000 "${flac_file%.*}.wav"
  rm "$flac_file"
done
