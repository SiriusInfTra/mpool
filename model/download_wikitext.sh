#!/bin/sh
unset https_proxy
export HF_ENDPOINT=https://hf-mirror.com
./hfd.sh wikitext --dataset --tool aria2c -x 4