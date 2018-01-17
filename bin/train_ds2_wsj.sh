#!/bin/sh
set -x
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:/usr/local/cuda-9.0/extras/CUPTI/lib64/:/usr/local/cuda-9.0/lib64/:$LD_LIBRARY_PATH

export COMPUTE_DATA_DIR=/data/speech/WSJ
# Warn if we can't find the train files
if [ ! -f "${COMPUTE_DATA_DIR}/wsj-train.csv" ]; then
    echo "Warning: It looks like you don't have the Switchboard corpus"       \
         "downloaded and preprocessed. Make sure \$COMPUTE_DATA_DIR points to the" \
         "folder where the Switchboard data is located, and that you ran the" \
         "importer script before running this script."
fi;

export EXPERIMENT=DS2-WSJ-F64-C3x2x32x64-R1x1024-B16x8

export LOG_DIR=/ds2/experiments/${EXPERIMENT}
export CHECKPOINT_DIR=/ds2/experiments/${EXPERIMENT}/checkpoints
export SUMMARY_DIR=/ds2/experiments/${EXPERIMENT}/summary

if [ ! -d "$LOG_DIR" ]; then
  mkdir  ${LOG_DIR}
fi
if [ ! -d "$CHECKPOINT_DIR" ]; then
  mkdir  ${CHECKPOINT_DIR}
fi
if [ ! -d "$SUMMARY_DIR" ]; then
  mkdir  ${SUMMARY_DIR}
fi

python -u DeepSpeech2.py \
  --train_files "${COMPUTE_DATA_DIR}/wsj-train.csv" \
  --dev_files "${COMPUTE_DATA_DIR}/wsj-dev.csv" \
  --test_files "${COMPUTE_DATA_DIR}/wsj-test.csv" \
  --num_mfcc 64 \
  --num_conv_layers 3 \
  --num_rnn_layers 1 \
  --rnn_cell_dim 1024 \
  --n_hidden 1024 \
  --train_batch_size 16 \
  --dev_batch_size  16 \
  --test_batch_size 16 \
  --epoch 100 \
  --early_stop 0 \
  --learning_rate 0.00005 \
  --display_step 0 \
  --validation_step 1 \
  --dropout_keep_prob 1.0 \
  --default_stddev 0.046875 \
  --checkpoint_dir "${CHECKPOINT_DIR}" \
  --checkpoint_secs 18000 \
  --wer_log_pattern "GLOBAL LOG: logwer('${COMPUTE_ID}', '%s', '%s', %f)"\
  --summary_dir  "${SUMMARY_DIR}" \
  --summary_secs 600 \
  --decoder_library_path /opt/tensorflow/bazel-bin/native_client/libctc_decoder_with_kenlm.so \
  --lm_binary_path data/lm/wsj-lm.binary \
  --lm_trie_path data/lm/wsj-lm.trie \
  --beam_width 64 \
  --word_count_weight 1.5 \
  --valid_word_count_weight 2.5 \
  "$@"  2>&1 | tee ${LOG_DIR}/${EXPERIMENT}_$(date +%Y%m%d_%H%M).txt
