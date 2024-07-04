#!/bin/sh
unset https_proxy
export HF_ENDPOINT=https://hf-mirror.com
./hfd.sh microsoft/phi-1.5 --tool aria2c -x 4