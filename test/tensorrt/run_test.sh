#!/bin/bash

script_dir=$(dirname "$(realpath "$0")")

echo "Output w/o mpool ------------------------------------------------"
output1=$(python $script_dir/test_tensorrt_infer.py)
echo ${output1}
result1=$(echo "$output1" | grep '^Result')


echo "Output w/ mpool ------------------------------------------------"
output2=$(python $script_dir/test_tensorrt_infer.py --use-mpool)
echo ${output2}
result2=$(echo "$output2" | grep '^Result')

if [ "$result1" == "$result2" ]; then
    echo "The results are the same."
else
    echo "The results are different."
fi
