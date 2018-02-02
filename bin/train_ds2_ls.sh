#!/bin/sh
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:/usr/local/cuda-9.0/extras/CUPTI/lib64/:/usr/local/cuda-9.0/lib64/:$LD_LIBRARY_PATH

export COMPUTE_DATA_DIR=/raid/DATA/SPEECH/LibriSpeech
# Warn if we can't find the train files
if [ ! -f "${COMPUTE_DATA_DIR}/wsj-train.csv" ]; then
    echo "Warning: It looks like you don't have the Switchboard corpus"       \
         "downloaded and preprocessed. Make sure \$COMPUTE_DATA_DIR points to the" \
         "folder where the Switchboard data is located, and that you ran the" \
         "importer script before running this script."
fi;

export EXPERIMENT=DS2-LS-F161-C32x64-R1x512-H512-B64x4-AUG-NT

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
  --validation_step 1 \
  --dropout_keep_prob 0.5 \
  --weight_decay 0.0005 \
  --checkpoint_dir ${CHECKPOINT_DIR} \
  --checkpoint_secs 18000 \
  --summary_dir ${SUMMARY_DIR} \
  --summary_secs 600 \
  --lm_binary_path data/lm/wsj-lm.binary \
  --lm_trie_path data/lm/wsj-lm.trie \
  --beam_width 64 \
  --word_count_weight 1.5 \
  --valid_word_count_weight 2.5 \
"

echo VERSION: $(git rev-parse --short HEAD) | tee $LOG_FILE
echo CONFIG: | tee -a $LOG_FILE
echo $CONFIG | tee -a $LOG_FILE

python -u DeepSpeech2.py $CONFIG \
  --wer_log_pattern "GLOBAL LOG: logwer('${COMPUTE_ID}', '%s', '%s', %f)" \
  --decoder_library_path /opt/tensorflow/bazel-bin/native_client/libctc_decoder_with_kenlm.so \
  "$@" 2>&1 | tee -a $LOG_FILE
