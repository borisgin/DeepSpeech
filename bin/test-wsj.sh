#!/bin/sh
set -xe
#export CUDA_VISIBLE_DEVICES=1
export COMPUTE_DATA_DIR=/data/speech/WSJ
export COMPUTE_KEEP_DIR=/ds2/WSJ-41-DS1

if [ ! -f DeepSpeech.py ]; then
    echo "Please make sure you run this from DeepSpeech's top level directory."
    exit 1
fi;


# Warn if we can't find the train files
if [ ! -f "${COMPUTE_DATA_DIR}/wsj-train.csv" ]; then
    echo "Warning: It looks like you don't have the Switchboard corpus"       \
         "downloaded and preprocessed. Make sure \$COMPUTE_DATA_DIR points to the" \
         "folder where the Switchboard data is located, and that you ran the" \
         "importer script before running this script."
fi;

if [ -d "${COMPUTE_KEEP_DIR}" ]; then
    checkpoint_dir=$COMPUTE_KEEP_DIR
else
    checkpoint_dir=$(python -c 'from xdg import BaseDirectory as xdg; print(xdg.save_data_path("deepspeech/swb"))')
fi


python -u DeepSpeech.py \
  --train_files "${COMPUTE_DATA_DIR}/wsj-train.csv" \
  --dev_files "${COMPUTE_DATA_DIR}/wsj-dev.csv" \
  --test_files "${COMPUTE_DATA_DIR}/wsj-test.csv" \
  --train_batch_size 16 \
  --dev_batch_size 16 \
  --test_batch_size 16 \
  --train 0 \
  --learning_rate 0.0001 \
  --checkpoint_dir "${COMPUTE_KEEP_DIR}" \
  --wer_log_pattern "GLOBAL LOG: logwer('${COMPUTE_ID}', '%s', '%s', %f)" \
  --decoder_library_path /opt/tensorflow/bazel-bin/native_client/libctc_decoder_with_kenlm.so \
  --lm_binary_path data/lm/wsj-lm.binary \
  --lm_trie_path data/lm/wsj-lm.trie \
  --word_count_weight 1.0 \
  --valid_word_count_weight 2.5 \
  "$@"
