#!/bin/bash
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:/usr/local/cuda-9.0/extras/CUPTI/lib64/:/usr/local/cuda-9.0/lib64/:$LD_LIBRARY_PATH

export LM_DIR=/data/speech/LM
export COMPUTE_DATA_DIR=/data/speech/LibriSpeech

export EXPERIMENT=DS2-LS-F161-C32x64-R1x512-H512-B64x8-AUG-NT

export LOG_DIR=/ds2/experiments/${EXPERIMENT}
export CHECKPOINT_DIR=${LOG_DIR}/checkpoints
export SUMMARY_DIR=${LOG_DIR}/summary

if [ ! -d "$LOG_DIR" ]; then
  mkdir  ${LOG_DIR}
fi
if [ ! -d "$CHECKPOINT_DIR" ]; then
  mkdir  ${CHECKPOINT_DIR}
fi
if [ ! -d "$SUMMARY_DIR" ]; then
  mkdir  ${SUMMARY_DIR}
fi

LOG_FILE=${LOG_DIR}/${EXPERIMENT}_$(date +%Y%m%d_%H%M).txt

echo Logging the experiment to $LOG_FILE


CONFIG="\
  --train_files $COMPUTE_DATA_DIR/librivox-train-clean-100.csv,$COMPUTE_DATA_DIR/librivox-train-clean-360.csv,$COMPUTE_DATA_DIR/librivox-train-other-500.csv \
  --dev_files $COMPUTE_DATA_DIR/librivox-dev-clean.csv,$COMPUTE_DATA_DIR/librivox-dev-other.csv \
  --test_files $COMPUTE_DATA_DIR/librivox-test-clean.csv,$COMPUTE_DATA_DIR/librivox-test-other.csv \
  --input_type spectrogram \
  --num_audio_features 161 \
  --num_conv_layers 2 \
  --num_rnn_layers 1 \
  --rnn_cell_dim 512 \
  --rnn_type gru \
  --n_hidden 512 \
  --train_batch_size 64 \
  --dev_batch_size  16 \
  --test_batch_size 16 \
  --epoch 100 \
  --early_stop 0 \
  --optimizer adam \
  --learning_rate 0.0001 \
  --decay_steps 3000 \
  --decay_rate 0.9 \
  --display_step 10 \
  --validation_step 10 \
  --dropout_keep_prob 0.5 \
  --weight_decay 0.0005 \
  --checkpoint_dir ${CHECKPOINT_DIR} \
  --checkpoint_secs 18000 \
  --summary_dir ${SUMMARY_DIR} \
  --summary_secs 600 \
  --lm_binary_path $LM_DIR/ls-n3-lm.binary \
  --lm_trie_path $LM_DIR/ls-n3-lm.trie \
  --beam_width 64 \
  --word_count_weight 1.5 \
  --valid_word_count_weight 2.5 \
"

echo VERSION: $(git rev-parse --short HEAD) | tee $LOG_FILE
echo CONFIG: | tee -a $LOG_FILE
echo $CONFIG | tee -a $LOG_FILE

time python -u DeepSpeech2.py $CONFIG \
  --wer_log_pattern "GLOBAL LOG: logwer('${COMPUTE_ID}', '%s', '%s', %f)" \
  --decoder_library_path /opt/tensorflow/bazel-bin/native_client/libctc_decoder_with_kenlm.so \
  "$@" 2>&1 | tee -a $LOG_FILE
