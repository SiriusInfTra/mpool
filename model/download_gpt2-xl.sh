#!/bin/sh
unset https_proxy
export HF_ENDPOINT=https://hf-mirror.com
./hfd.sh openai-community/gpt2-xl --tool aria2c -x 4