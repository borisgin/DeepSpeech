#!/bin/bash
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:/usr/local/cuda-9.0/extras/CUPTI/lib64/:/usr/local/cuda-9.0/lib64/:$LD_LIBRARY_PATH


export EXPERIMENT="TMP"
export LOG_DIR${EXPERIMENT}
export CHECKPOINT_DIR=${LOG_DIR}/checkpoints



if [ ! -d "$LOG_DIR" ]; then
  mkdir  ${LOG_DIR}
fi
if [ ! -d "$CHECKPOINT_DIR" ]; then
  mkdir  ${CHECKPOINT_DIR}
fi
if [ ! -d "$SUMMARY_DIR" ]; then
  mkdir  ${SUMMARY_DIR}
fi

export CUDA_VISIBLE_DEVICE=0,1,2,3,4,5,6,7

LOG_FILE=${LOG_DIR}/PERFTEST_${EXPERIMENT}_8GPU_$(date +%Y%m%d_%H%M).txt

echo Logging the experiment to $LOG_FILE



CONFIG="\
  --perf_seq_len=${SEQ_LEN} --perf_num_iters=${NUM_ITERS} \
  --input_type spectrogram --num_audio_features 161 \
  --num_conv_layers 3 \
  --rnn_unidirectional=True \
  --rnn_type cudnn_gru  --num_rnn_layers ${NUM_RNN_LAYERS}  --rnn_cell_dim ${HIDDEN_SIZE} \
  --row_conv=True --row_conv_width 8 \
  --n_hidden ${HIDDEN_SIZE} \
  --train_batch_size ${BATCH_SIZE} \
  --dev_batch_size ${BATCH_SIZE} \
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
  --test_batch_size $BATCH_SIZE \
  --wer_log_pattern "GLOBAL LOG: logwer('${COMPUTE_ID}', '%s', '%s', %f)" \
  --decoder_library_path GREEDY_DECODER \
  "$@" 2>&1 | tee -a $LOG_FILE

