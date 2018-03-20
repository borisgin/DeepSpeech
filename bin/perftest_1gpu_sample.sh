#!/bin/bash
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:/usr/local/cuda-9.0/extras/CUPTI/lib64/:/usr/local/cuda-9.0/lib64/:$LD_LIBRARY_PATH


export EXPERIMENT=./MODEL
export LOG_DIR=${EXPERIMENT}
export CHECKPOINT_DIR=${EXPERIMENT}



if [ ! -d "$LOG_DIR" ]; then
  mkdir  ${LOG_DIR}
fi
if [ ! -d "$CHECKPOINT_DIR" ]; then
  mkdir  ${CHECKPOINT_DIR}
fi

export CUDA_VISIBLE_DEVICES=0

LOG_FILE=${LOG_DIR}/VAL_TEST_1GPU_$(date +%Y%m%d_%H%M).txt

echo Logging the experiment to $LOG_FILE



CONFIG="\
  --train_files=data/sample/ldc93s1.csv \
  --dev_files=data/sample/ldc93s1.csv \
  --test_files=data/sample/ldc93s1.csv \
  --input_type spectrogram --num_audio_features 161 \
  --num_conv_layers 3 \
  --rnn_unidirectional=True \
  --rnn_type cudnn_gru  --num_rnn_layers 2  --rnn_cell_dim 1024 \
  --row_conv=True --row_conv_width 8 \
  --n_hidden 1024 \
  --train_batch_size 8 \
  --dev_batch_size 8 \
  --test_batch_size 8 \
  --train 0 \
  --learning_rate 0 \
  --display_step 1 \
  --validation_step 1 \
  --dropout_keep_prob 1.0 \
  --weight_decay 0 \
  --checkpoint_dir ${CHECKPOINT_DIR} \
  --device gpu \
"

echo VERSION: $(git rev-parse --short HEAD) | tee $LOG_FILE
echo CONFIG: | tee -a $LOG_FILE
echo $CONFIG | tee -a $LOG_FILE

time python -u DeepSpeech2.py $CONFIG \
  --wer_log_pattern "GLOBAL LOG: logwer('${COMPUTE_ID}', '%s', '%s', %f)" \
  --decoder_library_path GREEDY_DECODER \
  "$@" 2>&1 | tee -a $LOG_FILE

