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




nohup python3.7 run_infer.py --gpu=1 --nr_type=6 --type_info_path=type_info.json --batch_size=32 --model_mode=fast --model_path=checkpoint/hovernet_fast_pannuke_type_tf2pytorch.tar tile --input_dir=../datasets/image500/1102-1/ --output_dir=../output/pannuke/`date +%Y%m%d-%H:%M`_1102-1 --draw_dot>nohup3.out &