#!/bin/bash
####
BASE_DIR=
EXEC_FILE="$0"
BASE_NAME=`basename "$EXEC_FILE"`
if [ "$EXEC_FILE" = "./$BASE_NAME" ] || [ "$EXEC_FILE" = "$BASE_NAME" ]; then
        BASE_DIR=`pwd`
else
        BASE_DIR=`echo "$EXEC_FILE" | sed 's/'"\/${BASE_NAME}"'$//'`
fi
####
PARAM_NAME="$1"

### export path
export PYTHONPATH=/home/sjwang/lib
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

### experiment run




nohup python3.7 run_infer.py --gpu=0 --nr_types=6 --type_info_path=type_info.json --batch_size=32 --model_mode=fast --model_path=checkpoint/hovernet_fast_pannuke_type_tf2pytorch.tar wsi --input_dir=../datasets/WSI/WSI/ --output_dir=../output/pannuke_wsi --input_mask_dir=../datasets/WSI/mask/ --save_thumb --save_mask>nohup2.out &