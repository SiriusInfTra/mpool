#!/bin/sh
unset https_proxy
export HF_ENDPOINT=https://hf-mirror.com
./hfd.sh microsoft/phi-2 --tool aria2c -x 4