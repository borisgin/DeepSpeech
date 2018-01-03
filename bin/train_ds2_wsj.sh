#!/bin/sh
set -xe
export EXPERIMENT=DS2-WSJ-Cx32x32x96-R1x1024-B16

#export LD_LIBRARY_PATH=/usr/local/cuda-9.0/extras/CUPTI/lib64/:/usr/local/cuda-9.0/lib64/:$LD_LIBRARY_PATH
export COMPUTE_DATA_DIR=/data/speech/WSJ
export CHECKPOINT_DIR=/ds2/experiments/${EXPERIMENT}/checkpoints
export SUMMARY_DIR=/ds2/experiments/${EXPERIMENT}/summary
export LOG_DIR=/ds2/experiments/${EXPERIMENT}

# Warn if we can't find the train files
if [ ! -f "${COMPUTE_DATA_DIR}/wsj-train.csv" ]; then
    echo "Warning: It looks like you don't have the Switchboard corpus"       \
         "downloaded and preprocessed. Make sure \$COMPUTE_DATA_DIR points to the" \
         "folder where the Switchboard data is located, and that you ran the" \
         "importer script before running this script."
fi;

python -u DeepSpeech2.py \
  --train_files "${COMPUTE_DATA_DIR}/wsj-train.csv" \
  --dev_files "${COMPUTE_DATA_DIR}/wsj-dev.csv" \
  --test_files "${COMPUTE_DATA_DIR}/wsj-test.csv" \
  --n_hidden 1024 \
  --num_rnn_layers 1 \  
  --train_batch_size 16 \
  --dev_batch_size 16 \
  --test_batch_size 16 \
  --epoch 15 \
  --early_stop 0 \
  --learning_rate 0.0001 \
  --display_step 1 \
  --validation_step 1 \
  --dropout_rate 0.15 \
  --default_stddev 0.046875 \
  --checkpoint_dir "${CHECKPOINT_DIR}" \
  --checkpoint_secs 18000 \
  --wer_log_pattern "GLOBAL LOG: logwer('${COMPUTE_ID}', '%s', '%s', %f)"\
  --summary_dir  "${SUMMARY_DIR}" \
  --summary_secs 600 \
  --decoder_library_path /opt/tensorflow/bazel-bin/native_client/libctc_decoder_with_kenlm.so \
  "$@"  2>&1 | tee ${LOG_DIR}/${EXPERIMENT}_$(date +%Y%m%d_%H%M).txt
